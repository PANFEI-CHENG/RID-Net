# Reference: https://github.com/haoyu94/RoITr
import torch
import torch.nn as nn
import numpy as np
from model.model import *
from model.modules import CoarseMatching, FineMatching, LearnableLogOptimalTransport
import torch.nn.functional as F
from lib.utils import point_to_node_partition, index_select

class RID_Net(nn.Module):

    def __init__(self, config):
        super(RID_Net, self).__init__()

        # backbone network
        self.coarse_matching = CoarseMatching(num_correspondences=config.num_est_coarse_corr, dual_normalization=True)
        self.factor = 1
        self.backbone = AMR_FPN(config, transformer_architecture=config.transformer_architecture, factor=self.factor)
        # learnable Optimal Transport Layer
        self.OT = LearnableLogOptimalTransport(num_iter=100)
        # the number of correspondences used for each point cloud pair during training

        self.point_per_patch = config.point_per_patch
        
        # coarse level final descriptor projection
        self.coarse_proj = nn.Linear(256*self.factor, 256*self.factor)
        # fine level final descriptor projection
        self.fine_proj = nn.Linear(64*self.factor, 256*self.factor)

        self.fine_matching = FineMatching(config.fine_matching_topk,
                                          mutual=config.fine_matching_mutual, confidence_threshold=config.fine_matching_confidence_threshold,
                                          use_dustbin=config.fine_matching_use_dustbin,
                                          use_global_score=config.fine_matching_use_global_score,
                                          correspondence_threshold=config.fine_matching_correspondence_threshold)

        self.fine_matching_use_dustbin = config.fine_matching_use_dustbin

        self.optimal_transport = LearnableLogOptimalTransport(num_iter=100)
    
    def forward(self, src_points, tgt_points, src_normals, tgt_normals, src_feats, tgt_feats, src_v, tgt_v):
        src_o, tgt_o = torch.from_numpy(np.array([src_points.shape[0]])).to(src_points).int(), torch.from_numpy(np.array([tgt_points.shape[0]])).to(tgt_points).int()
        output_dict = {}
        # 1. get descriptors
        src_node_xyz, src_node_feats, src_points, src_point_feats, tgt_node_xyz, tgt_node_feats, tgt_points, tgt_point_feats, \
            src_node_n, tgt_node_n = self.backbone([src_points, src_feats, src_o, src_normals, src_v], [tgt_points, tgt_feats, tgt_o, tgt_normals, tgt_v])

        src_node_feats = F.normalize(self.coarse_proj(src_node_feats), p=2, dim=1)
        tgt_node_feats = F.normalize(self.coarse_proj(tgt_node_feats), p=2, dim=1)

        src_point_feats = self.fine_proj(src_point_feats)
        tgt_point_feats = self.fine_proj(tgt_point_feats)

        # 2. get ground truth node correspondences
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(src_points, src_node_xyz, point_limit=self.point_per_patch)
        _, tgt_node_masks, tgt_node_knn_indices, tgt_node_knn_masks = point_to_node_partition(tgt_points, tgt_node_xyz, point_limit=self.point_per_patch)


        src_padded_points = torch.cat([src_points, torch.zeros_like(src_points[:1])], dim=0)
        tgt_padded_points = torch.cat([tgt_points, torch.zeros_like(tgt_points[:1])], dim=0)
        src_node_knn_points = index_select(src_padded_points, src_node_knn_indices, dim=0)
        tgt_node_knn_points = index_select(tgt_padded_points, tgt_node_knn_indices, dim=0)

        # 3. select topk node correspondences
        with torch.no_grad():

            tgt_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(tgt_node_feats, src_node_feats, tgt_node_masks, src_node_masks)
            output_dict['src_node_corr_indices'] = src_node_corr_indices
            output_dict['tgt_node_corr_indices'] = tgt_node_corr_indices

        # 4. Generate batched node points & feats
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        tgt_node_corr_knn_indices = tgt_node_knn_indices[tgt_node_corr_indices]  # (P, K)

        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        tgt_node_corr_knn_masks = tgt_node_knn_masks[tgt_node_corr_indices]  # (P, K)

        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)
        tgt_node_corr_knn_points = tgt_node_knn_points[tgt_node_corr_indices]  # (P, K, 3)

        src_padded_point_feats = torch.cat([src_point_feats, torch.zeros_like(src_point_feats[:1])], dim=0)
        tgt_padded_point_feats = torch.cat([tgt_point_feats, torch.zeros_like(tgt_point_feats[:1])], dim=0)

        src_node_corr_knn_feats = index_select(src_padded_point_feats, src_node_corr_knn_indices, dim=0)  # (P, K, C)
        tgt_node_corr_knn_feats = index_select(tgt_padded_point_feats, tgt_node_corr_knn_indices, dim=0)  # (P, K, C)

        # 5. Optimal transport

        matching_scores = torch.einsum('bnd,bmd->bnm', tgt_node_corr_knn_feats,
                                       src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / src_point_feats.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, tgt_node_corr_knn_masks, src_node_corr_knn_masks)

        # 6. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching_use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            tgt_corr_points, src_corr_points, corr_scores = self.fine_matching(
                tgt_node_corr_knn_points,
                src_node_corr_knn_points,
                tgt_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            output_dict['tgt_corr_points'] = tgt_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores

        return output_dict

def create_model(config):
    model = RID_Net(config)
    return model