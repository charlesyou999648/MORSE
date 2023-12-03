import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from networks.net_factory_args import net_factory
from networks.net_factory_3d_args import net_factory_3d, UNet3D
from point_rend.semantic_seg_3d import PointRendSemSegHead,  calculate_uncertainty
from point_rend.point_head_3d import StandardPointHead, ImplicitPointHead
from mmcv.cnn import ConvModule
from point_rend.point_features_3d import (
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from mmcv.cnn import build_conv_layer
import random

def create_model(num_classes=4):
    # Network definition
    model = net_factory(net_type='unet', in_chns=1,
                        class_num=num_classes)
    return model


def create_model_3d(num_classes=4):
    # Network definition
    # model = net_factory_3d(net_type="vnet", in_chns=1,
    #                     class_num=num_classes)
    model = UNet3D(in_channels=1, out_channels=num_classes)
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
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for affine in self.affine_layers:
            x = affine(x)
            if affine != self.affine_layers[-1] or self.final_act:
                x = self.activation(x)
        x = x.transpose(1, 2).reshape(B, -1, D, H, W)
        return x


class MOEHead_3d(nn.Module):
    def __init__(self, fea_dim=[128, 64, 32, 16, 16], output_dim=128) -> None:
        super().__init__()
        self.in_channels = fea_dim
        self.output_dim = output_dim
        self.prescale_mlp_dims = [64, 64]
        self.prescale_mlp = nn.ModuleList()
        for in_channel in self.in_channels: # [64, 128, 320, 512]
            mlp = MLP(in_channel, self.prescale_mlp_dims, final_act=True, activation='relu')
            self.prescale_mlp.append(mlp)
        cur_dim = len(self.in_channels) * self.prescale_mlp_dims[-1]
        self.moe_conv = nn.ModuleList()
        conv_dims = [output_dim] + [len(self.in_channels)]
        
        for conv_dim in conv_dims:
            conv_layer = nn.Sequential(
                nn.Conv3d(in_channels=cur_dim, 
                          out_channels=conv_dim,
                          kernel_size=3, 
                          stride=1, 
                          padding=1),
                nn.BatchNorm3d(conv_dim), 
                nn.ReLU(inplace=True),
                )
            cur_dim = conv_dim
            self.moe_conv.append(conv_layer)
        afterscale_mlp_dims = [32, 32]
        self.afterscale_mlp = MLP(self.prescale_mlp_dims[-1], afterscale_mlp_dims, final_act=True, activation='relu')
        cur_dim = afterscale_mlp_dims[-1]
        
        dropout_ratio=0.1
        if dropout_ratio > 0:
            self.dropout = nn.Dropout3d(dropout_ratio)
        self.linear_pred = nn.Conv3d(cur_dim, self.output_dim, kernel_size=1)
    
    def forward(self, x):
        largest_size = x[-1].shape[-3:]
        x_scaled = []
        for i, x_i in enumerate(x):
            if self.prescale_mlp_dims is not None:
                x_i = self.prescale_mlp[i](x_i)

            if x_i.shape[-3:] != largest_size:
                x_i_scaled = F.interpolate(x_i, size=largest_size, mode='trilinear', align_corners=False)
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
        
        
        
class MOEHead_3d_UseOne(nn.Module):
    def __init__(self, fea_dim=[16, 16, 16], output_dim=128, use_sparse=False) -> None:
        super().__init__()
        self.in_channels = fea_dim
        self.output_dim = output_dim
        self.prescale_mlp_dims = [64, 64]
        self.prescale_mlp = nn.ModuleList()
        for in_channel in self.in_channels: # [64, 128, 320, 512]
            mlp = MLP(in_channel, self.prescale_mlp_dims, final_act=True, activation='relu')
            self.prescale_mlp.append(mlp)
        cur_dim = len(self.in_channels) * self.prescale_mlp_dims[-1]
        self.moe_conv = nn.ModuleList()
        conv_dims = [output_dim] + [len(self.in_channels)]
        
        for conv_dim in conv_dims:
            conv_layer = nn.Sequential(
                nn.Conv3d(in_channels=cur_dim, 
                          out_channels=conv_dim,
                          kernel_size=3, 
                          stride=1, 
                          padding=1),
                nn.BatchNorm3d(conv_dim), 
                nn.ReLU(inplace=True),
                )
            cur_dim = conv_dim
            self.moe_conv.append(conv_layer)
        afterscale_mlp_dims = [32, 32]
        self.afterscale_mlp = MLP(self.prescale_mlp_dims[-1], afterscale_mlp_dims, final_act=True, activation='relu')
        cur_dim = afterscale_mlp_dims[-1]
        
        dropout_ratio=0.1
        if dropout_ratio > 0:
            self.dropout = nn.Dropout3d(dropout_ratio)
        self.linear_pred = nn.Conv3d(cur_dim, self.output_dim, kernel_size=1)
        self.use_sparse = use_sparse
        self.use_num = len(fea_dim) - 1
    
    def forward(self, x):
        largest_size = x[-1].shape[-3:]
        x_scaled = []
        for i, x_i in enumerate(x):
            if self.prescale_mlp_dims is not None:
                x_i = self.prescale_mlp[i](x_i)

            if x_i.shape[-3:] != largest_size:
                x_i_scaled = F.interpolate(x_i, size=largest_size, mode='trilinear', align_corners=False)
            else:
                x_i_scaled = x_i

            x_scaled.append(x_i_scaled)
        # 256, 9
        x_stacked = torch.stack(x_scaled, dim=1) 
        x = torch.cat(x_scaled, dim=1)
        
        for conv_layer in self.moe_conv:
            x = conv_layer(x)
        moe_weights = torch.softmax(x, dim=1)
        
        if self.use_sparse and self.training:
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
    

class FeatureExtractor_3d(nn.Module):
    #### can be modified as moe ####
    def __init__(self, fea_dim=[128, 64, 32, 16, 16], output_dim=128) -> None:
        super().__init__()
        # assert len(fea_dim)==5, 'input_dim is not correct'
        cnt = fea_dim[0]
        self.features = nn.ModuleList()
        for i in range(len(fea_dim)-1):
            self.features.append(nn.Conv3d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False))
            cnt += fea_dim[i+1]
        
        self.features.append(nn.Conv3d(in_channels=cnt, out_channels=output_dim, kernel_size=1, bias=False))
        
    def forward(self, fea_list):
        assert len(fea_list) == len(self.features)
        x = fea_list[0]
        for i, feat in enumerate(fea_list[:-1]):
            x = self.features[i](x) + x
            x = nn.Upsample(size=fea_list[i+1].shape[-3:], mode='trilinear', align_corners=True)(x)
            x = torch.cat([x, fea_list[i+1]], dim=1)
        x = self.features[-1](x)
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
    

class PointRend(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = create_model_3d(num_classes=args.num_classes)
        if args.moe:
            # self.feature_extractor = MOEHead_3d(fea_dim=[32, 16, 16], output_dim=args.in_features)
            self.feature_extractor = MOEHead_3d_UseOne(fea_dim=[16, 16, 16], \
                output_dim=args.in_features)
        else:
            self.feature_extractor = FeatureExtractor_3d(fea_dim=[32, 16, 16], output_dim=args.in_features)
        # if args.diffusion:
        #     self.pointrend_seg_head = PointRendSemSegHeadDiffusion(args)
        # else:
        self.pointrend_seg_head = PointRendSemSegHead(args)
    
    def forward(self, image, label=None):
        pred, featmap = self.model(image)
        # print([item.shape for item in featmap])
        # torch.Size([2, 32, 56, 56, 40]), torch.Size([2, 16, 112, 112, 80]), torch.Size([2, 16, 112, 112, 80])]
        # featmap = self.feature_extractor(featmap[-3:])
        featmap = self.feature_extractor(3*featmap[-1:])
        res = self.pointrend_seg_head(pred_logits=pred, features=featmap, targets=label)
        return pred, res


class Patcher_UseOne(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = create_model_3d(num_classes=args.num_classes)
        self.feature_extractor = MOEHead_3d_UseOne(fea_dim=[16, 16, 16], \
            output_dim=args.num_classes, use_sparse=args.moe_random)
    
    def forward(self, image, label=None):
        _, featmap = self.model(image)
        res = self.feature_extractor(3*featmap[-1:])
        return res, featmap
    
    
class Patcher(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = create_model_3d(num_classes=args.num_classes)
        self.feature_extractor = MOEHead_3d(fea_dim=[64, 32, 16, 16], \
            output_dim=args.num_classes, use_sparse=args.moe_random)
    
    def forward(self, image, label=None):
        _, featmap = self.model(image)
        featmap = self.feature_extractor(featmap[-4:])
        return featmap
    
    
class PointRend_patcher(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = Patcher_UseOne(args)
        self.feature_extractor = FeatureExtractor_3d(fea_dim=[32, 16, 16], output_dim=args.in_features)
        self.pointrend_seg_head = PointRendSemSegHead(args)
    
    def forward(self, image, label=None):
        pred, featmap = self.model(image)
        # print([item.shape for item in featmap])
        # torch.Size([2, 32, 56, 56, 40]), torch.Size([2, 16, 112, 112, 80]), torch.Size([2, 16, 112, 112, 80])]
        featmap = self.feature_extractor(featmap[-3:])
        # featmap = self.feature_extractor(3*featmap[-1:])
        res = self.pointrend_seg_head(pred_logits=pred, features=featmap, targets=label)
        return pred, res
    
    
   