import argparse, os
from configs.utils import load_config, normal_redirect, cal_v
import torch
from easydict import EasyDict as edict
import os.path as osp
import open3d as o3d
import numpy as np
from model.RID_Net import create_model
from lib.utils import ransac_pose_estimation_correspondences, release_cuda

def Visualization(data, n_points=5000):
    src_corr_pts, tgt_corr_pts = data['src_corr_points'], data['tgt_corr_points']
    src_raw_o3d = data['src_raw_o3d']
    tgt_raw_o3d = data['tgt_raw_o3d']
    confidence = data['corr_scores']

    if confidence.shape[0] > n_points:
            sel_idx = torch.topk(torch.from_numpy(confidence), k=n_points)[1]
            src_corr_pts, tgt_corr_pts = src_corr_pts[sel_idx], tgt_corr_pts[sel_idx]
            confidence = confidence[sel_idx]
    correspondences = torch.from_numpy(np.arange(src_corr_pts.shape[0])).unsqueeze(1).repeat(1, 2)
    tsfm_est=ransac_pose_estimation_correspondences(src_corr_pts, tgt_corr_pts, correspondences)
    src_color = np.full((len(src_raw_o3d.points), 3), [0, 0.651, 0.929])#蓝色
    tgt_color = np.full((len(tgt_raw_o3d.points), 3), [1, 0.706, 0])#黄色
    src_raw_o3d.colors = o3d.utility.Vector3dVector(src_color)
    tgt_raw_o3d.colors = o3d.utility.Vector3dVector(tgt_color)

    o3d.visualization.draw_geometries([src_raw_o3d,tgt_raw_o3d])
    o3d.visualization.draw_geometries([src_raw_o3d.transform(tsfm_est),tgt_raw_o3d])

def Data_processing(src_path, tgt_path, cfg, points_lim=30000, view_point=np.array([0., 0., 0.])):
    src_raw_o3d = o3d.io.read_point_cloud(src_path)
    tgt_raw_o3d = o3d.io.read_point_cloud(tgt_path)
    src_pcd = src_raw_o3d.voxel_down_sample(0.025)
    tgt_pcd = tgt_raw_o3d.voxel_down_sample(0.025)
    src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=33))
    src_pcd.normalize_normals()

    tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=33))
    tgt_pcd.normalize_normals()

    src_points = np.asarray(src_pcd.points)
    src_normals = np.asarray(src_pcd.normals)
    src_normals = normal_redirect(src_points, src_normals, view_point)

    tgt_points = np.asarray(tgt_pcd.points)
    tgt_normals = np.asarray(tgt_pcd.normals)
    tgt_normals = normal_redirect(tgt_points, tgt_normals, view_point)

    src_v = cal_v(src_normals, src_points)
    tgt_v = cal_v(tgt_normals, tgt_points)

    if src_points.shape[0] > points_lim:
        idx = np.random.permutation(src_points.shape[0])[:points_lim]
        src_points = src_points[idx]
        src_normals = src_normals[idx]
        src_v = src_v[idx]
    if tgt_points.shape[0] > points_lim:
        idx = np.random.permutation(tgt_points.shape[0])[:points_lim]
        tgt_points = tgt_points[idx]
        tgt_normals = tgt_normals[idx]
        tgt_v = tgt_v[idx]
    src_feats = np.ones(shape=(src_points.shape[0], 1))
    tgt_feats = np.ones(shape=(tgt_points.shape[0], 1))

    return  (src_points.astype(np.float32), tgt_points.astype(np.float32), \
            src_normals.astype(np.float32), tgt_normals.astype(np.float32),\
            src_feats.astype(np.float32), tgt_feats.astype(np.float32),\
            src_v.astype(np.float32), tgt_v.astype(np.float32)),\
            src_raw_o3d, tgt_raw_o3d,

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    config = load_config(args.config)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    else:
        raise('cuda is not available')
    config['output_dir'] = 'results/{}'.format(config['exp_dir'])
    config = edict(config)
    os.makedirs(config.output_dir, exist_ok=True)
    print('Starting to process the input...')
     
    input_model, src_raw_o3d, tgt_raw_o3d = Data_processing(osp.join(config.root, 'src.ply'), osp.join(config.root, 'tgt.ply'), config)
    input_model = tuple(torch.from_numpy(item).cuda() for item in input_model)
    print('Data processing completed.')

    model = create_model(config).cuda()
    state_dict = torch.load(config.pretrain)
    model.load_state_dict(state_dict)
    print('Starting registration...')
    output = model(*input_model)
    output = release_cuda(output)
    output['src_raw_o3d'] = src_raw_o3d
    output['tgt_raw_o3d'] = tgt_raw_o3d

    ##Visualization
    Visualization(output)


if __name__ == '__main__':
    main()