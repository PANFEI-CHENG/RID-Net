import yaml
import numpy as np
import open3d as o3d
import torch

def load_config(path):
    '''
    Load config file
    :param path: path to config file
    :return: a dictionary consisting of loaded configuration, sub dictionaries will be merged
    '''
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for key, value in cfg.items():
        for k, v in value.items():
            config[k] = v

    return config

def normal_redirect(points, normals, view_point):
    '''
    Make direction of normals towards the view point
    '''
    vec_dot = np.sum((view_point - points) * normals, axis=-1)
    mask = (vec_dot < 0.)
    redirected_normals = normals.copy()
    redirected_normals[mask] *= -1.
    return redirected_normals

def cal_v(normals, pcd):
    zp = np.expand_dims(normals, 1)#N,1,3

    knn_search = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(pcd, dtype=o3d.core.Dtype.Float32))
    query_points = o3d.core.Tensor(pcd, dtype=o3d.core.Dtype.Float32)
    knn_search.knn_index()
    indices, distances = knn_search.knn_search(query_points, knn=33)#distances是距离的平方

    distances = np.sqrt(distances.numpy())
    x = query_points[indices[:,1:]].numpy() - np.expand_dims(query_points.numpy(), axis=1)#N,32,3
    x = x.transpose(0, 2, 1)#N,3,32
    norm = (zp @ x).transpose(0, 2, 1)#N,33,1
    proj = norm * zp #N,33,3
    vi = x - proj.transpose(0, 2, 1)
    x_l2 = distances[:,1:]

    exponent  = np.floor(np.log10(np.abs(distances[:,-1])))
    significant_digit = distances[:,-1] / (10 ** exponent)
    significant_digit += 0.5
    r = np.expand_dims(significant_digit * (10 ** exponent), 1)#N

    alpha = (r - x_l2)/(r + 1e-8)
    alpha = alpha * alpha
    norm = norm / (r[:,:,np.newaxis] + 1e-8)
    beta = (norm * norm).transpose(0, 2, 1)

    alpha = np.where(alpha < 1e-6, np.ones_like(alpha)*1e-6, alpha)
    beta = np.where(beta < 1e-6, np.ones_like(beta)*1e-6, beta)

    vi_c = (alpha[:,np.newaxis,:] * beta * vi).sum(2)   
    xpp = (vi_c / (np.sqrt((vi_c ** 2).sum(1, keepdims=True)) + 1e-8))     

    return xpp

# def to_device(x, device):

#     if isinstance(x, list):
#         x = [to_device(item, device) for item in x]
#     elif isinstance(x, tuple):
#         x = (to_device(item, device) for item in x)
#     elif isinstance(x, dict):
#         x = {key: to_device(value, device) for key, value in x.items()}
#     elif isinstance(x, torch.Tensor):
#         # x = x.cuda()
#         x = x.to(device)
    
#     return x