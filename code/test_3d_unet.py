import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import shutil
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from networks.net_factory_args import net_factory
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage import zoom
from model_3d import Patcher, Patcher_UseOne, create_model_3d
from dataloaders.dataset_3d import *
from torchvision.utils import make_grid

from medpy import metric
import h5py
from torch.utils.data import Dataset
import os
import argparse
import torch
from networks.vnet import VNet
import math
import nibabel as nib
import torch.nn.functional as F
# from model_3d_semi import PointRend_semi
# from model_3d import PointRend
from model_3d import Patcher, Patcher_UseOne, create_model_3d, PointRend
# from test_util import test_all_case

def test_all_case(net, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None):
    total_metric = 0.0
    for image_path in tqdm(image_list):
        id = image_path.split('/')[-1]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:].transpose(1, 2, 0)
        label = h5f['label'][:].transpose(1, 2, 0)
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        metric_list = []
        for i in range(1, num_classes):
            single_metric = calculate_metric_percase(prediction==i, label[:]==i)
                
            metric_list.append(single_metric)
            
        print(metric_list)
        total_metric += np.asarray(metric_list)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    return avg_metric


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch)#.cuda()
                with torch.no_grad():
                    y1 = net(test_patch)[0]
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction==i)
        label_tmp = (label==i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:

        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, jc, hd95, asd
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 1, 0, 0
    else: 
        return 0, 0, 0, 0

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/chenyu/Documents/UA-MT/data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='LA/point_rend_semi_8_labeled/unet', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--epoch', type=int, default=30000)
parser.add_argument('--fc_dim', type=int, default=256)
parser.add_argument('--num_fc', type=int, default=3)
parser.add_argument('--input_channels', type=int, default=256)
parser.add_argument('--coarse_pred_each_layer', type=eval, default=True, choices=[True, False])
parser.add_argument('--cls_agnostic_mask', type=eval, default=False, choices=[True, False])
parser.add_argument('--moe', type=eval, default=False, choices=[True, False])
parser.add_argument('--moe_random', type=eval, default=False, choices=[True, False])

parser.add_argument('--train_num_points', type=int, default=2048)
parser.add_argument('--in_features', type=int, default=256)
parser.add_argument('--oversample_ratio', type=float, default=3)
parser.add_argument('--importance_sample_ratio', type=float, default=0.75)
parser.add_argument('--subdivision_steps', type=int, default=2)
parser.add_argument('--subdivision_num_points', type=int, default=8192)
parser.add_argument('--implicit', type=eval, default=False, choices=[True, False])
parser.add_argument('--diffusion_loss_ratio', type=float, default=0.1)
parser.add_argument('--diffusion_loss_type', type=str,
                    default='mse', help='mse or l1')
parser.add_argument('--batch_size', type=int, default=4,
                help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2,
                help='labeled_batch_size per gpu')
FLAGS = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/"+FLAGS.model+"/"
test_save_path = "../model/prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = FLAGS.num_classes

with open(FLAGS.root_path + '/../test.txt', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path+'/'+item.replace('\n','')+'.h5' for item in image_list]


def test_calculate_metric(epoch_num):
    # net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    # net = create_model_3d(num_classes=num_classes).cuda()
    net = PointRend(args=FLAGS)#.cuda()
    
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    net.load_state_dict(torch.load(save_mode_path, map_location='cpu'))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    metric_ = test_calculate_metric(FLAGS.epoch)
    print(metric_)
    print(sum(metric_)/len(metric_))