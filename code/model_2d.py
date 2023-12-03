import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import argparse
from networks.net_factory_args import net_factory
from networks.net_factory_3d_args import net_factory_3d
from point_rend.semantic_seg\
    import PointRendSemSegHead, calculate_uncertainty, \
    ConvFCHead, ImplicitPointHead_origin
from point_rend.point_head import StandardPointHead, ImplicitPointHead
from mmcv.cnn import ConvModule
from point_rend.point_features import (
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import numpy as np
from mmcv.cnn import build_conv_layer
from network import VIT_net
    
    
def create_model(num_classes=4):
    # Network definition
    model = net_factory(net_type='unet', in_chns=1,
                        class_num=num_classes)
    return model


def create_model_3d(num_classes=4):
    # Network definition
    model = net_factory_3d(net_type="vnet", in_chns=1,
                        class_num=num_classes)
    return model


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), final_act=True, activation='relu'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.final_act = final_act
        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for affine in self.affine_layers:
            x = affine(x)
            if affine != self.affine_layers[-1] or self.final_act:
                x = self.activation(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        return x


class MOEHead(nn.Module):
    def __init__(self, fea_dim=[256, 128, 64, 32, 16], output_dim=256) -> None:
        super().__init__()
        self.in_channels = fea_dim
        self.output_dim = output_dim
        self.prescale_mlp_dims = [256, 256]
        self.prescale_mlp = nn.ModuleList()
        for in_channel in self.in_channels: # [64, 128, 320, 512]
            mlp = MLP(in_channel, self.prescale_mlp_dims, final_act=True, activation='relu')
            self.prescale_mlp.append(mlp)
        cur_dim = len(self.in_channels) * self.prescale_mlp_dims[-1]
        self.moe_conv = nn.ModuleList()
        conv_dims = [output_dim] + [len(self.in_channels)]
        
        for conv_dim in conv_dims:
            conv_layer = ConvModule(
                in_channels=cur_dim,
                out_channels=conv_dim,
                kernel_size=3, stride=1, padding=1,
                norm_cfg=dict(type='BN', requires_grad=True)
            )
            cur_dim = conv_dim
            self.moe_conv.append(conv_layer)
        afterscale_mlp_dims = [256, 256]
        self.afterscale_mlp = MLP(self.prescale_mlp_dims[-1], afterscale_mlp_dims, final_act=True, activation='relu')
        cur_dim = afterscale_mlp_dims[-1]
        
        dropout_ratio=0.1
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(cur_dim, self.output_dim, kernel_size=1)
    
    def forward(self, x):
        largest_size = x[-1].shape[-2:]
        x_scaled = []
        for i, x_i in enumerate(x):
            if self.prescale_mlp_dims is not None:
                x_i = self.prescale_mlp[i](x_i)

            if x_i.shape[-2:] != largest_size:
                x_i_scaled = F.interpolate(x_i, size=largest_size, mode='bilinear', align_corners=False)
            else:
                x_i_scaled = x_i

            x_scaled.append(x_i_scaled)
        # 256, 9
        x_stacked = torch.stack(x_scaled, dim=1) 
        x = torch.cat(x_scaled, dim=1)
        
        for conv_layer in self.moe_conv:
            x = conv_layer(x)
        moe_weights = torch.softmax(x, dim=1)
        
        x = (x_stacked * moe_weights.unsqueeze(2)).sum(1)
        x = self.afterscale_mlp(x)
        x = self.dropout(x)
        x = self.linear_pred(x)
        
        return x
    
        

class FeatureExtractor(nn.Module):
    #### can be modified as moe ####
    def __init__(self, fea_dim=[256, 128, 64, 32, 16], output_dim=256) -> None:
        super().__init__()
        # assert len(fea_dim)==5, 'input_dim is not correct'
        cnt = fea_dim[0]
        self.features = nn.ModuleList()
        for i in range(len(fea_dim)-1):
            self.features.append(nn.Conv2d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False))
            cnt += fea_dim[i+1]
        self.features.append(nn.Conv2d(in_channels=cnt, out_channels=output_dim, kernel_size=1, bias=False))
        
    def forward(self, fea_list):
        assert len(fea_list) == len(self.features)
        x = fea_list[0]
        for i, feat in enumerate(fea_list[:-1]):
            x = self.features[i](x) + x
            x = nn.Upsample(size=fea_list[i+1].shape[-2:], mode='bilinear', align_corners=True)(x)
            x = torch.cat([x, fea_list[i+1]], dim=1)
        x = self.features[-1](x)
        return x
    
    
class PointRend_Transunet(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = VIT_net(num_class=args.num_classes)
        if args.moe_random:
            self.feature_extractor = MOEHead_random(output_dim=args.in_features,fea_dim=[512, 256, 32])
            self.feature_extractor.use_num = 2
        elif args.moe:
            self.feature_extractor = MOEHead(output_dim=args.in_features,fea_dim=[512, 256, 32])
        else:
            self.feature_extractor = FeatureExtractor(output_dim=args.in_features,fea_dim=[512, 256, 32])
        self.pointrend_seg_head = PointRendSemSegHead(args)
    
    def forward(self, image, label=None):
        pred, featmap = self.model(image)
        featmap = featmap[:-1] 
        # print([item.shape for item in featmap])
        # [torch.Size([2, 512, 32, 32]), torch.Size([2, 256, 64, 64]), torch.Size([2, 64, 128, 128]), torch.Size([2, 768, 16, 16])]
        featmap = self.feature_extractor(featmap)
        res = self.pointrend_seg_head(pred_logits=pred, features=featmap, targets=label)
        return pred, res


class MOEHead_random(nn.Module):
    def __init__(self, fea_dim=[256, 128, 64, 32, 16], output_dim=256) -> None:
        super().__init__()
        self.in_channels = fea_dim
        self.output_dim = output_dim
        self.prescale_mlp_dims = [256, 256]
        self.prescale_mlp = nn.ModuleList()
        for in_channel in self.in_channels: # [64, 128, 320, 512]
            mlp = MLP(in_channel, self.prescale_mlp_dims, final_act=True, activation='relu')
            self.prescale_mlp.append(mlp)
        cur_dim = len(self.in_channels) * self.prescale_mlp_dims[-1]
        self.moe_conv = nn.ModuleList()
        conv_dims = [output_dim] + [len(self.in_channels)]
        
        for conv_dim in conv_dims:
            conv_layer = ConvModule(
                in_channels=cur_dim,
                out_channels=conv_dim,
                kernel_size=3, stride=1, padding=1,
                norm_cfg=dict(type='BN', requires_grad=True)
            )
            cur_dim = conv_dim
            self.moe_conv.append(conv_layer)
        afterscale_mlp_dims = [256, 256]
        self.afterscale_mlp = MLP(self.prescale_mlp_dims[-1], afterscale_mlp_dims, final_act=True, activation='relu')
        cur_dim = afterscale_mlp_dims[-1]
        
        dropout_ratio=0.1
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(cur_dim, self.output_dim, kernel_size=1)
        self.use_num = 4
    
    def forward(self, x):
        largest_size = x[-1].shape[-2:]
        x_scaled = []
        for i, x_i in enumerate(x):
            if self.prescale_mlp_dims is not None:
                x_i = self.prescale_mlp[i](x_i)

            if x_i.shape[-2:] != largest_size:
                x_i_scaled = F.interpolate(x_i, size=largest_size, mode='bilinear', align_corners=False)
            else:
                x_i_scaled = x_i

            x_scaled.append(x_i_scaled)
        # 256, 9
        x_stacked = torch.stack(x_scaled, dim=1) 
        x = torch.cat(x_scaled, dim=1)
        
        for conv_layer in self.moe_conv:
            x = conv_layer(x)
        moe_weights = torch.softmax(x, dim=1)
        
        # print('moe_weights', moe_weights.shape) # moe_weights torch.Size([2, 5, 256, 256])
        if self.training:
            ret = random.randint(0, self.use_num)
            un_used_num = random.sample(range(moe_weights.shape[1]), ret)
            modified = moe_weights.clone()
            for item in un_used_num:
                modified[:, item] = 0
        else:
            modified = moe_weights.clone()
        
        x = (x_stacked * modified.unsqueeze(2)).sum(1)
        x = self.afterscale_mlp(x)
        x = self.dropout(x)
        x = self.linear_pred(x)
        
        return x

class PointRend(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = create_model(num_classes=args.num_classes)
        if args.moe_random:
            self.feature_extractor = MOEHead_random(output_dim=args.in_features)
        elif args.moe:
            self.feature_extractor = MOEHead(output_dim=args.in_features)
        else:
            self.feature_extractor = FeatureExtractor(output_dim=args.in_features)
        self.pointrend_seg_head = PointRendSemSegHead(args)
    
    def forward(self, image, label=None):
        pred, _, featmap = self.model(image)
        featmap = self.feature_extractor(featmap)
        res = self.pointrend_seg_head(pred_logits=pred, features=featmap, targets=label)
        return pred, res

    
class PointRendSepMoE(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = create_model(num_classes=args.num_classes)
        self.train_num_points        = args.train_num_points        # 2048
        self.oversample_ratio        = args.oversample_ratio        # 3
        self.importance_sample_ratio = args.importance_sample_ratio # 0.75
        self.subdivision_num_points  = args.subdivision_num_points  # 8192
        self.implicit                = args.implicit
        self.num_classes             = args.num_classes
        args.cls_agnostic_mask        = True
        # args.num_classes = 32
        self.in_channels = [256, 128, 64, 32, 16]
        self.output_dim = self.num_classes
        self.prescale_mlp_dims = [256, 256]
        self.prescale_mlp = nn.ModuleList()
        for in_channel in self.in_channels: # [64, 128, 320, 512]
            mlp = MLP(in_channel, self.prescale_mlp_dims, final_act=True, activation='relu')
            self.prescale_mlp.append(mlp)
        cur_dim = len(self.in_channels) * self.prescale_mlp_dims[-1]
        # cur_dim = len(self.in_channels) * 32
        self.moe_conv = nn.ModuleList()
        conv_dims = [self.output_dim] + [len(self.in_channels)]
        for conv_dim in conv_dims:
            conv_layer = nn.Conv1d(in_channels=cur_dim, 
                                   out_channels=conv_dim, 
                                   kernel_size=1)
            BN_layer = nn.BatchNorm1d(num_features=conv_dim)
            ReLu_layer = nn.ReLU(inplace=True)
            cur_dim = conv_dim
            self.moe_conv.append(conv_layer)
            self.moe_conv.append(BN_layer)
            self.moe_conv.append(ReLu_layer)
        afterscale_mlp_dims = [256, 256]
        self.afterscale_mlp_dims = afterscale_mlp_dims
        self.afterscale_mlp = nn.Sequential(
            nn.Conv1d(afterscale_mlp_dims[0], afterscale_mlp_dims[0], kernel_size=1), 
            nn.ReLU(inplace=True), 
            nn.Conv1d(afterscale_mlp_dims[0], afterscale_mlp_dims[1], kernel_size=1), 
            nn.ReLU(inplace=True), 
        )
        # self.afterscale_mlp = MLP(self.prescale_mlp_dims[-1], afterscale_mlp_dims, final_act=True, activation='relu')
        cur_dim = afterscale_mlp_dims[-1]
        
        dropout_ratio=0.1
        if dropout_ratio > 0:
            self.dropout = nn.Dropout1d(dropout_ratio)
        self.linear_pred = nn.Conv1d(cur_dim, self.output_dim, kernel_size=1)
        if self.implicit:
            self.point_heads = nn.ModuleList()
            for in_channel in self.in_channels:
                self.point_heads.append(ImplicitPointHead(args))
        else:
            self.point_heads = nn.ModuleList()
            for in_channel in self.in_channels:
                self.point_heads.append(StandardPointHead(args))
        
        
    def forward(self, image, targets=None):
        if self.training:
            pred_logits, _, x = self.model(image)
            largest_size = x[-1].shape[-2:]
            x_scaled = []
            for i, x_i in enumerate(x):
                if self.prescale_mlp_dims is not None:
                    x_i = self.prescale_mlp[i](x_i)
                if x_i.shape[-2:] != largest_size:
                    x_i_scaled = F.interpolate(x_i, size=largest_size, mode='bilinear', align_corners=False)
                else:
                    x_i_scaled = x_i
                x_scaled.append(x_i_scaled)
            # print([item.shape for item in x_scaled]) [5*(2, 256, 256, 256)]
            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    pred_logits,
                    calculate_uncertainty,
                    self.train_num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                # print(point_coords)
            coarse_features = point_sample(pred_logits, point_coords, align_corners=False)
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
            loss = 0.0
            x_after_rend = []
            for index, features in enumerate(x_scaled):
                fine_grained_features = point_sample(features, point_coords, align_corners=False)
                point_logits = self.point_heads[index](fine_grained_features, coarse_features)
                # want this to be 32
                x_after_rend.append(point_logits)
                
            # print([item.shape for item in x_after_rend])
            x_stack = torch.stack(x_after_rend, dim=1)
            x = torch.cat(x_after_rend, dim=1)
            
            # print('x_stack', x_stack.shape) # x_stack torch.Size([2, 5, 32, 2048])
            # print('x', x.shape) # x torch.Size([2, 160, 2048])
            
            for conv_layer in self.moe_conv:
                x = conv_layer(x)
            moe_weights = torch.softmax(x, dim=1)
            
            x = (x_stack * moe_weights.unsqueeze(2)).sum(1)
            # print(x.shape) # torch.Size([2, 32, 2048])
            x = self.afterscale_mlp(x)
            x = self.dropout(x)
            x = self.linear_pred(x)
            loss = F.cross_entropy(x, point_targets)
                # loss += F.cross_entropy(
                #     point_logits, point_targets, reduction="mean"
                # )*(self.in_channels[-index]/sum(self.in_channels))
            return pred_logits, loss
        else:
            pred_logits, _, x = self.model(image)
            sem_seg_logits = pred_logits.clone()
            largest_size = x[-1].shape[-2:]
            x_scaled = []
            for i, x_i in enumerate(x):
                if self.prescale_mlp_dims is not None:
                    x_i = self.prescale_mlp[i](x_i)
                if x_i.shape[-2:] != largest_size:
                    x_i_scaled = F.interpolate(x_i, size=largest_size, mode='bilinear', align_corners=False)
                else:
                    x_i_scaled = x_i
                x_scaled.append(x_i_scaled)
            uncertainty_map = calculate_uncertainty(sem_seg_logits)
            point_indices, point_coords = get_uncertain_point_coords_on_grid(
                uncertainty_map, self.subdivision_num_points
            )
            coarse_features = point_sample(pred_logits, point_coords, align_corners=False)
            x_after_rend = []
            for index, features in enumerate(x_scaled):
                fine_grained_features = point_sample(features, point_coords, align_corners=False)
                point_logits = self.point_heads[index](fine_grained_features, coarse_features)
                # want this to be 32
                x_after_rend.append(point_logits)
            x_stack = torch.stack(x_after_rend, dim=1)
            x = torch.cat(x_after_rend, dim=1)
            
            # print('x_stack', x_stack.shape) # x_stack torch.Size([2, 5, 32, 2048])
            # print('x', x.shape) # x torch.Size([2, 160, 2048])
            
            for conv_layer in self.moe_conv:
                x = conv_layer(x)
            moe_weights = torch.softmax(x, dim=1)
            
            x = (x_stack * moe_weights.unsqueeze(2)).sum(1)
            # print(x.shape) # torch.Size([2, 32, 2048])
            x = self.afterscale_mlp(x)
            x = self.dropout(x)
            x = self.linear_pred(x)
            N, C, H, W = sem_seg_logits.shape
            point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
            sem_seg_logits = (
                sem_seg_logits.reshape(N, C, H * W)
                .scatter_(2, point_indices, x)
                .view(N, C, H, W)
            )
            return pred_logits, sem_seg_logits
            


class PointRend_implicit(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = create_model(num_classes=args.num_classes)
        if args.moe_random:
            self.feature_extractor = MOEHead_random(output_dim=args.in_features)
        elif args.moe:
            self.feature_extractor = MOEHead(output_dim=args.in_features)
        else:
            self.feature_extractor = FeatureExtractor(output_dim=args.in_features)
        self.pointrend_seg_head = PointRendSemSegHead(args)
    
    def forward(self, image, label=None):
        pred, _, featmap = self.model(image)
        featmap = self.feature_extractor(featmap)
        res = self.pointrend_seg_head(pred_logits=pred, features=featmap, targets=label)
        return pred, res

