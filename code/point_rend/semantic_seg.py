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

from .point_features import (
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from .point_head import StandardPointHead, ImplicitPointHead

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
    # print('calculate_uncertainty', ans.shape) # calculate_uncertainty torch.Size([1, 1, 6144])
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
        # features shape: (B_s, channel, h, w)
        # pred_logits shape: (B_s, class, h, w)
        # targets shape: (B_s, h, w)
        if self.training:
            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    pred_logits,
                    calculate_uncertainty,
                    self.train_num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                # print('point_coords', point_coords.shape) # (1, 2048, 2)
            coarse_features = point_sample(pred_logits, point_coords, align_corners=False)
            fine_grained_features = point_sample(features, point_coords, align_corners=False)
            if not self.implicit:
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
                #     sem_seg_logits, scale_factor=2, mode="bilinear", align_corners=False
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
                N, C, H, W = sem_seg_logits.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                sem_seg_logits = (
                    sem_seg_logits.reshape(N, C, H * W)
                    .scatter_(2, point_indices, point_logits)
                    .view(N, C, H, W)
                )
            return sem_seg_logits
        
        
                  
class ImplicitPointHead_origin(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained features and instance-wise MLP parameters as its input.
    """

    def __init__(self, cfg):
        """
        The following attributes are parsed from config:
            channels: the output dimension of each FC layers
            num_layers: the number of FC layers (including the final prediction layer)
            image_feature_enabled: if True, fine-grained image-level features are used
            positional_encoding_enabled: if True, positional encoding is used
        """
        super(ImplicitPointHead_origin, self).__init__()
        # fmt: off
        self.num_layers                         = cfg.num_fc  + 1
        self.channels                           = cfg.fc_dim 
        self.image_feature_enabled              = True
        self.positional_encoding_enabled        = True
        self.num_classes                        = cfg.num_classes 
        self.in_channels                        = cfg.input_channels   
        # fmt: on

        if not self.image_feature_enabled:
            self.in_channels = 0
        if self.positional_encoding_enabled:
            self.in_channels += 256
            self.register_buffer("positional_encoding_gaussian_matrix", torch.randn((2, 128)))

        assert self.in_channels > 0

        num_weight_params, num_bias_params = [], []
        assert self.num_layers >= 2
        for l in range(self.num_layers):
            if l == 0:
                # input layer
                num_weight_params.append(self.in_channels * self.channels)
                num_bias_params.append(self.channels)
            elif l == self.num_layers - 1:
                # output layer
                num_weight_params.append(self.channels * self.num_classes)
                num_bias_params.append(self.num_classes)
            else:
                # intermediate layer
                num_weight_params.append(self.channels * self.channels)
                num_bias_params.append(self.channels)

        self.num_weight_params = num_weight_params
        self.num_bias_params = num_bias_params
        self.num_params = sum(num_weight_params) + sum(num_bias_params)

    def forward(self, fine_grained_features, point_coords, parameters):
        # features: [R, channels, K]
        # point_coords: [R, K, 2]
        num_instances = fine_grained_features.size(0)
        num_points = fine_grained_features.size(2)

        if num_instances == 0:
            return torch.zeros((0, 1, num_points), device=fine_grained_features.device)

        if self.positional_encoding_enabled:
            # locations: [R*K, 2]
            locations = 2 * point_coords.reshape(num_instances * num_points, 2) - 1
            locations = locations @ self.positional_encoding_gaussian_matrix.to(locations.device)
            locations = 2 * np.pi * locations
            locations = torch.cat([torch.sin(locations), torch.cos(locations)], dim=1)
            # locations: [R, C, K]
            locations = locations.reshape(num_instances, num_points, 256).permute(0, 2, 1)
            if not self.image_feature_enabled:
                fine_grained_features = locations
            else:
                fine_grained_features = torch.cat([locations, fine_grained_features], dim=1)

        # features [R, C, K]
        mask_feat = fine_grained_features.reshape(num_instances, self.in_channels, num_points)

        weights, biases = self._parse_params(
            parameters,
            self.in_channels,
            self.channels,
            self.num_classes,
            self.num_weight_params,
            self.num_bias_params,
        )

        point_logits = self._dynamic_mlp(mask_feat, weights, biases, num_instances)
        point_logits = point_logits.reshape(-1, self.num_classes, num_points)

        return point_logits

    @staticmethod
    def _dynamic_mlp(features, weights, biases, num_instances):
        assert features.dim() == 3, features.dim()
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = torch.einsum("nck,ndc->ndk", x, w) + b
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    @staticmethod
    def _parse_params(
        pred_params,
        in_channels,
        channels,
        num_classes,
        num_weight_params,
        num_bias_params,
    ):
        assert pred_params.dim() == 2
        assert len(num_weight_params) == len(num_bias_params)
        assert pred_params.size(1) == sum(num_weight_params) + sum(num_bias_params), sum(num_weight_params) + sum(num_bias_params)

        num_instances = pred_params.size(0)
        num_layers = len(num_weight_params)

        params_splits = list(
            torch.split_with_sizes(pred_params, num_weight_params + num_bias_params, dim=1)
        )

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l == 0:
                # input layer
                weight_splits[l] = weight_splits[l].reshape(num_instances, channels, in_channels)
                bias_splits[l] = bias_splits[l].reshape(num_instances, channels, 1)
            elif l < num_layers - 1:
                # intermediate layer
                weight_splits[l] = weight_splits[l].reshape(num_instances, channels, channels)
                bias_splits[l] = bias_splits[l].reshape(num_instances, channels, 1)
            else:
                # output layer
                weight_splits[l] = weight_splits[l].reshape(num_instances, num_classes, channels)
                bias_splits[l] = bias_splits[l].reshape(num_instances, num_classes, 1)

        return weight_splits, bias_splits


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("")
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--fc_dim', type=int, default=256)
    parser.add_argument('--num_fc', type=int, default=3)
    parser.add_argument('--input_channels', type=int, default=256)
    parser.add_argument('--coarse_pred_each_layer', type=eval, default=True, choices=[True, False])
    parser.add_argument('--cls_agnostic_mask', type=eval, default=False, choices=[True, False])
    
    parser.add_argument('--train_num_points', type=int, default=2048)
    parser.add_argument('--in_features', type=int, default=256)
    parser.add_argument('--oversample_ratio', type=float, default=3)
    parser.add_argument('--importance_sample_ratio', type=float, default=1.0)
    parser.add_argument('--subdivision_steps', type=int, default=2)
    parser.add_argument('--subdivision_num_points', type=int, default=8192)
    parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
    parser.add_argument('--diffusion_loss_ratio', type=float, default=0.1)
    parser.add_argument('--diffusion_loss_type', type=str,
                    default='mse', help='mse or l1')
    parser.add_argument('--implicit', type=eval, default=False, choices=[True, False])
    
    args = parser.parse_args()
    
    # model = PointRendSemSegHead(args)
    # # model = PointRendSemSegHead(args)
    # model.eval()
    # inp = torch.randn(1, 9, 256, 256)
    # feat = torch.randn(1, 256, 256, 256)
    # target = torch.ones(1, 256, 256).long()
    
    # res = model(inp, feat, target) 
    # print(res.shape) # tensor(2.2055, grad_fn=<NllLoss2DBackward0>) # torch.Size([2, 9, 1024, 1024])
    
    # res, point_logits = model.ddim_sample(inp, feat) # 
    # print(res.shape) # ()
    # print(point_logits.shape) # torch.Size([2, 9, 8192])
    
    model = ImplicitPointHead_origin(args)
    print(model.num_params) # 265225