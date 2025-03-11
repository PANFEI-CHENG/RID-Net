# Reference: https://github.com/qinzheng93/GeoTransformer

import torch.nn as nn
from model.transformer.positional_encoding import PPFStructualEmbedding, GeometricStructureEmbedding
from model.transformer.attention import LocalRPEAttentionLayer


class LocalPPFTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        dropout=None,
    ):
        r"""Geometric Transformer (GeoTransformer).
        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            activation_fn: activation function
        """
        super(LocalPPFTransformer, self).__init__()

        self.embedding = PPFStructualEmbedding(hidden_dim)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = LocalRPEAttentionLayer(d_model=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        feats,
        node_idx,
        group_idx,
        ppfs
    ):
        r"""Geometric Transformer
        Args:
            feats (Tensor): (N, in_dim)
            node_idx: (M,)
            group_idx: (M, K)
            ppfs (Tensor): (M, K, 4)
        Returns:
            new_feats: torch.Tensor (M, C2)
        """
        pos_embeddings = self.embedding(ppfs) #[M, K, hidden_dims]
        feats = self.in_proj(feats) #[N, in_dim] -> [N, hidden_dim]
        new_feats, _ = self.transformer(
            feats,
            pos_embeddings,
            node_idx,
            group_idx
        )
        new_feats = self.out_proj(new_feats)

        return new_feats
