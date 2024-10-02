import torch
import numpy as np
import open3d as o3d

def release_cuda(x):
    r"""Release all tensors to item or numpy array."""
    if isinstance(x, list):
        x = [release_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (release_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu().numpy()
    return x

def to_o3d_pcd(pcd):
    '''
    Transfer a point cloud of numpy.ndarray to open3d point cloud
    :param pcd: point cloud of numpy.ndarray in shape[N, 3]
    :return: open3d.geometry.PointCloud()
    '''
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pcd)
    return pcd_

def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor
    
def get_inlier_ratio_correspondence(src_node, tgt_node, rot, trans, inlier_distance_threshold=0.1):
    '''
    Compute inlier ratios based on input torch tensors
    '''
    src_node = torch.matmul(src_node, rot.T) + trans.T
    dist = torch.norm(src_node - tgt_node, dim=-1)
    inliers = dist < inlier_distance_threshold
    inliers_num = torch.sum(inliers)
    return inliers_num / src_node.shape[0], inliers

def ransac_pose_estimation_correspondences(src_pcd, tgt_pcd, correspondences, mutual=False, distance_threshold=0.05,
                                           ransac_n=3):
    '''
    Run RANSAC estimation based on input correspondences
    :param src_pcd:
    :param tgt_pcd:
    :param correspondences:
    :param mutual:
    :param distance_threshold:
    :param ransac_n:
    :return:
    '''

    # ransac_n = correspondences.shape[0]

    if mutual:
        raise NotImplementedError
    else:
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        correspondences = o3d.utility.Vector2iVector(to_array(correspondences))

        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(src_pcd, tgt_pcd,
                                                                                               correspondences,
                                                                                               distance_threshold,
                                                                                               o3d.pipelines.registration.TransformationEstimationPointToPoint(
                                                                                                   False), ransac_n, [
                                                                                                   o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                                                                                                       0.9),
                                                                                                   o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                                                                                                       distance_threshold)],
                                                                                               o3d.pipelines.registration.RANSACConvergenceCriteria(
                                                                                                   50000, 1000))
        '''
        result_ransac = o3d.registration.registration_ransac_based_on_correspondence(
            source=src_pcd, target=tgt_pcd, corres=correspondences,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            criteria=o3d.registration.RANSACConvergenceCriteria(50000, 1000))
        '''
    return result_ransac.transformation

def index_select(data: torch.Tensor, index: torch.LongTensor, dim: int) -> torch.Tensor:
    r"""Advanced index select.
    Returns a tensor `output` which indexes the `data` tensor along dimension `dim`
    using the entries in `index` which is a `LongTensor`.
    Different from `torch.index_select`, `index` does not has to be 1-D. The `dim`-th
    dimension of `data` will be expanded to the number of dimensions in `index`.
    For example, suppose the shape `data` is $(a_0, a_1, ..., a_{n-1})$, the shape of `index` is
    $(b_0, b_1, ..., b_{m-1})$, and `dim` is $i$, then `output` is $(n+m-1)$-d tensor, whose shape is
    $(a_0, ..., a_{i-1}, b_0, b_1, ..., b_{m-1}, a_{i+1}, ..., a_{n-1})$.
    Args:
        data (Tensor): (a_0, a_1, ..., a_{n-1})
        index (LongTensor): (b_0, b_1, ..., b_{m-1})
        dim: int
    Returns:
        output (Tensor): (a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})
    """
    output = data.index_select(dim, index.view(-1))

    if index.ndim > 1:
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)

    return output

def point_to_node_partition(
    points: torch.Tensor,
    nodes: torch.Tensor,
    point_limit: int,
    return_count: bool = False,
):
    r"""Point-to-Node partition to the point cloud.
    Fixed knn bug.
    Args:
        points (Tensor): (N, 3)
        nodes (Tensor): (M, 3)
        point_limit (int): max number of points to each node
        return_count (bool=False): whether to return `node_sizes`
    Returns:
        point_to_node (Tensor): (N,)
        node_sizes (LongTensor): (M,)
        node_masks (BoolTensor): (M,)
        node_knn_indices (LongTensor): (M, K)
        node_knn_masks (BoolTensor) (M, K)
    """
    sq_dist_mat = square_distance(nodes[None, ::], points[None, ::])[0]  # (M, N)

    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    node_masks = torch.zeros(nodes.shape[0], dtype=torch.bool).cuda()  # (M,)
    node_masks.index_fill_(0, point_to_node, True)

    matching_masks = torch.zeros_like(sq_dist_mat, dtype=torch.bool)  # (M, N)
    point_indices = torch.arange(points.shape[0]).cuda()  # (N,)
    matching_masks[point_to_node, point_indices] = True  # (M, N)
    sq_dist_mat.masked_fill_(~matching_masks, 1e12)  # (M, N)

    node_knn_indices = sq_dist_mat.topk(k=point_limit, dim=1, largest=False)[1]  # (M, K)
    node_knn_node_indices = index_select(point_to_node, node_knn_indices, dim=0)  # (M, K)
    node_indices = torch.arange(nodes.shape[0]).cuda().unsqueeze(1).expand(-1, point_limit)  # (M, K)
    node_knn_masks = torch.eq(node_knn_node_indices, node_indices)  # (M, K)
    node_knn_indices.masked_fill_(~node_knn_masks, points.shape[0])

    if return_count:
        unique_indices, unique_counts = torch.unique(point_to_node, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long).cuda()  # (M,)
        node_sizes.index_put_([unique_indices], unique_counts)
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks
    
def square_distance(src, tgt, normalized=False):
    '''
    Calculate Euclidean distance between every two points, for batched point clouds in torch.tensor
    :param src: source point cloud in shape [B, N, 3]
    :param tgt: target point cloud in shape [B, M, 3]
    :return: Squared Euclidean distance matrix in torch.tensor of shape[B, N, M]
    '''
    B, N, _ = src.shape
    _, M, _ = tgt.shape
    if normalized:
        dist = 2.0 - 2.0 * torch.matmul(src, tgt.permute(0, 2, 1).contiguous())
    else:
        dist = -2. * torch.matmul(src, tgt.permute(0, 2, 1).contiguous())
        dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
        dist += torch.sum(tgt ** 2, dim=-1).unsqueeze(-2)

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist

def cart2sphere(points, point_normals, patches, xpp, r):

    points = torch.unsqueeze(points, dim=1).expand(-1, patches.shape[1], -1)
    point_normals = torch.unsqueeze(point_normals, dim=1).expand(-1, patches.shape[1], -1)
    vec_d = patches - points #[n, n_samples, 3]
    # d = torch.sqrt(torch.sum(vec_d ** 2, dim=-1, keepdim=True)) / (r + 1e-6) #[n, n_samples, 1]
    d = torch.sqrt(torch.sum(vec_d ** 2, dim=-1, keepdim=True))  #[n, n_samples, 1]
    
    # angle(n1, vec_d)
    y = torch.sum(point_normals * vec_d, dim=-1, keepdim=True)
    x = torch.cross(point_normals, vec_d, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))

    vi = vec_d - y * point_normals

    xpp = torch.unsqueeze(xpp, dim=1).expand(-1, patches.shape[1], -1)
    y = torch.sum(xpp * vi, dim=-1, keepdim=True) #cos*|vi|
    x = torch.cross(xpp, vi, dim=-1)#dsin*|vi|
    x = torch.norm(x, p=2, dim=-1, keepdim=True)#sinxpp*|vi|

    ypp = torch.cross(point_normals, xpp, dim=-1)
    signal = torch.sum(ypp * vi, dim=-1, keepdim=True)#cosypp*|vi|
    signal = torch.where(signal >= 0, torch.tensor(1, dtype=signal.dtype).to(signal), torch.tensor(-1, dtype=signal.dtype).to(signal))
   
    x = signal * x 
    epsilon = 1e-10
    near_zeros = y.abs() < epsilon
    y = y * (near_zeros.logical_not())
    y = y + (near_zeros * epsilon)
    theta = (torch.atan2(x, y)) / np.pi #-180-180 -1-1
    theta = theta + (theta < 0) * 2
    return theta

def group_all(feats):
    '''
    all-to-all grouping
    feats: [n, c]
    out: grouped feat: [n, n, c]
    '''
    grouped_feat = torch.unsqueeze(feats, dim=0)
    grouped_feat = grouped_feat.expand(feats.shape[0], -1, -1) #[n, n, c]
    return grouped_feat


def calc_ppf_gpu(points, point_normals, patches, patch_normals):
    '''
    Calculate ppf gpu
    points: [n, 3]
    point_normals: [n, 3]
    patches: [n, nsamples, 3]
    patch_normals: [n, nsamples, 3]
    '''
    points = torch.unsqueeze(points, dim=1).expand(-1, patches.shape[1], -1)
    point_normals = torch.unsqueeze(point_normals, dim=1).expand(-1, patches.shape[1], -1)
    vec_d = patches - points #[n, n_samples, 3]
    d = torch.sqrt(torch.sum(vec_d ** 2, dim=-1, keepdim=True)) #[n, n_samples, 1]
    # angle(n1, vec_d)
    y = torch.sum(point_normals * vec_d, dim=-1, keepdim=True)
    x = torch.cross(point_normals, vec_d, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle1 = torch.atan2(x, y) / np.pi

    # angle(n2, vec_d)
    y = torch.sum(patch_normals * vec_d, dim=-1, keepdim=True)
    x = torch.cross(patch_normals, vec_d, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle2 = torch.atan2(x, y) / np.pi

    # angle(n1, n2)
    y = torch.sum(point_normals * patch_normals, dim=-1, keepdim=True)
    x = torch.cross(point_normals, patch_normals, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle3 = torch.atan2(x, y) / np.pi

    ppf = torch.cat([d, angle1, angle2, angle3], dim=-1) #[n, samples, 4]
    return ppf