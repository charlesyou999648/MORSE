# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F
import math
import copy
from collections import namedtuple
from detectron2.layers import ShapeSpec, cat
# from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .point_features_3d import (
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from .point_head_3d import StandardPointHead, ImplicitPointHead
# import torch
# from chamferdist import ChamferDistance

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def calculate_uncertainty(sem_seg_logits):
    """
    For each location of the prediction `sem_seg_logits` we estimate uncerainty as the
        difference between top first and top second predicted logits.

    Args:
        mask_logits (Tensor): A tensor of shape (N, C, ...), where N is the minibatch size and
            C is the number of foreground classes. The values are logits.

    Returns:
        scores (Tensor): A tensor of shape (N, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    top2_scores = torch.topk(sem_seg_logits, k=2, dim=1)[0]
    ans = (top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1)
    # print('calculate_uncertainty', ans.shape) # calculate_uncertainty torch.Size([1, 1, 6144, 1, 1])
    return ans

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class PointRendSemSegHead(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.train_num_points = args.train_num_points               # 2048
        self.in_features = args.in_features                         # 256
        self.oversample_ratio        = args.oversample_ratio        # 3
        self.importance_sample_ratio = args.importance_sample_ratio # 0.75
        self.subdivision_steps       = args.subdivision_steps       # 2
        self.subdivision_num_points  = args.subdivision_num_points  # 8192
        self.implicit                = args.implicit
        if self.implicit:
            self.point_head = ImplicitPointHead(args)
        else:
            self.point_head = StandardPointHead(args)
        
        
    def forward(self, pred_logits, features, targets=None):
        # features shape: (B_s, channel, h, w, d)
        # pred_logits shape: (B_s, class, h, w, d)
        # targets shape: (B_s, h, w, d)
        if self.training:
            
            # pass
            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    pred_logits,
                    calculate_uncertainty,
                    self.train_num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                # print('point_coords', point_coords.shape) # point_coords torch.Size([1, 2048, 1, 1, 3])
            coarse_features = point_sample(pred_logits, point_coords, align_corners=False)
            fine_grained_features = point_sample(features, point_coords, align_corners=False)
            if not self.implicit:
                # print('fine_grained_features', fine_grained_features.shape)
                # print('coarse_features', coarse_features.shape)
                point_logits = self.point_head(fine_grained_features, coarse_features)
            else:
                point_logits = self.point_head(fine_grained_features, coarse_features, point_coords)
            point_targets = (
                point_sample(
                    targets.unsqueeze(1).to(torch.float),
                    point_coords,
                    mode="nearest",
                    align_corners=False,
                )
                .squeeze(1)
                .to(torch.long)
            )
            loss = F.cross_entropy(
                point_logits, point_targets, reduction="mean"
            )
            return loss
            
        else:
            # pass
            sem_seg_logits = pred_logits.clone()
            for _ in range(self.subdivision_steps):
                # sem_seg_logits = F.interpolate(
                #     sem_seg_logits, scale_factor=2, mode="trilinear", align_corners=False
                # )
                uncertainty_map = calculate_uncertainty(sem_seg_logits)
                point_indices, point_coords = get_uncertain_point_coords_on_grid(
                    uncertainty_map, self.subdivision_num_points
                )
                fine_grained_features = point_sample(features, point_coords, align_corners=False)
                coarse_features = point_sample(
                    pred_logits, point_coords, align_corners=False
                )
                if not self.implicit:
                    point_logits = self.point_head(fine_grained_features, coarse_features)
                else:
                    point_logits = self.point_head(fine_grained_features, coarse_features, point_coords)
                    
                N, C, D, H, W = sem_seg_logits.shape
                
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                sem_seg_logits = (
                    sem_seg_logits.reshape(N, C, D* H * W)
                    .scatter_(2, point_indices, point_logits)
                    .view(N, C, D, H, W)
                )
            return sem_seg_logits
        
        

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

