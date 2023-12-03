from torch.utils.data.dataset import Dataset
from PIL import Image
from PIL import ImageFilter
import pandas as pd
import numpy as np
import torch
import os
import random
import itertools
import glob
from scipy.ndimage import zoom

import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
from torch.utils.data.sampler import Sampler
import h5py


class BaseDataSetsWithIndex_NoAug(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None, index=16, label_type=0):
        self._base_dir = base_dir
        self.index = index
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
            if(label_type==1):
                self.sample_list = self.sample_list[:index]
            else:
                self.sample_list = self.sample_list[index:]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num-index]
        print("total {} samples".format(len(self.sample_list)))
        self.output_size = [256, 256]
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        
        sample = {'image': image, 'label': label}
        # if self.split == "train" and self.transform!=None:
        #     sample = self.transform(sample)
        sample["idx"] = idx
        return sample



class BaseDataSetsWithIndex(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None, index=16, label_type=0):
        self._base_dir = base_dir
        self.index = index
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
            if(label_type==1):
                self.sample_list = self.sample_list[:index]
            else:
                self.sample_list = self.sample_list[index:]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num-index]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train" and self.transform!=None:
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        if 'Lits' in list_dir:
            self.sample_list = open(os.path.join(list_dir, self.split+'_40.txt')).readlines()
        
        elif (split == "test" or split == 'val'):
            self.sample_list = open(os.path.join(list_dir, self.split+'_vol.txt')).readlines()
        else:
            self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            # print(data_path)
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

class Synapse_datasetWithIndex(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, index=221, label_type=1):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'_40.txt')).readlines()
        self.data_dir = base_dir
        self.index = index
        self.label_type = label_type
        if(label_type==1):
            self.sample_list = self.sample_list[:index]
        else:
            self.sample_list = self.sample_list[index:]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            # print(data_path)
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
    
    
    
    
    

if __name__ == '__main__':
    # dataset = Synapse_dataset(base_dir='/home/chenyu/Documents/Segmentation/project_TransUNet_JHU/data/JHU/train_npz_3class', 
    # list_dir='/home/chenyu/Documents/Segmentation/project_TransUNet_JHU/data/JHU', split='train')
    dataset = BaseDataSetsWithIndex()
    print(len(dataset)) # 5675
    data = dataset[0]
    print(data.keys()) # dict_keys(['image', 'label', 'case_name'])
    print(data['image'].shape) # (256, 256)
    print(data['label'].shape) # (256, 256)