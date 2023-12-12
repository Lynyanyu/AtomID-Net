import torch
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import csv
from PIL import Image
import pandas as pd
from pathlib import Path


PATH = "./datasets"

# Dataset
class ImageLoader(Dataset):
    def __init__(self, data_root, file_root, mode:str="train", crop_size:int=256, is_syn:bool=False):
        self.img_path = []
        self.labels = []
        self.names = []
        self.mode = mode
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5], [0.5])

        if mode=="train":
            if is_syn:
                data = pd.read_csv(os.path.join(file_root, 'tem_unet_syn_train.csv'))
            else:
                data = pd.read_csv(os.path.join(file_root, 'tem_unet_train.csv'))
            self.crop_size = crop_size
        else:
            data = pd.read_csv(os.path.join(file_root, 'tem_unet_test.csv'))
            self.crop_size = -1

        for i in range(len(data)):
            img_name = os.path.basename(data["image"].iloc[i]).split('.')[0]
            img_path = Path(data_root, data["image"].iloc[i])
            csv_path = Path(data_root, data["csv"].iloc[i])
            self.names.append(str(img_name))
            self.img_path.append(str(img_path))
            self.labels.append(str(csv_path))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = self.img_path[index]
        csv_path = self.labels[index]
        
        with open(img_path, 'rb') as fm:
            sample = Image.open(fm).convert('L')
            sample = self.toTensor(sample)
        
        label = torch.zeros(sample.shape)
        resize_ratio = 1.0
        if self.mode == "train":
            '''Random Resize'''
            resize_ratio = np.random.randint(6, 11) / 8.0
            label = F.interpolate(label.unsqueeze(0), scale_factor=resize_ratio).squeeze(0)
            sample = F.interpolate(sample.unsqueeze(0), scale_factor=resize_ratio).squeeze(0)
        
        with open(csv_path, 'r') as csv_file:
            content = csv.reader(csv_file)
            next(content)
            for line in content:
                xcoo = min(round(float(line[2])*resize_ratio), label.size(1)-1)
                ycoo = min(round(float(line[1])*resize_ratio), label.size(2)-1)
                label[:, xcoo, ycoo] = 1.0
        
        sample, label = self.preprocess(sample, label)
        return sample, label, self.names[index]
        
    
    def preprocess(self, sample, label):

        '''Random Crop'''
        if self.crop_size > 0:
            crop_anchor_h = np.random.randint(0, sample.size(1)-self.crop_size)
            crop_anchor_w = np.random.randint(0, sample.size(2)-self.crop_size)
            sample = sample[:, crop_anchor_h:crop_anchor_h+self.crop_size, crop_anchor_w:crop_anchor_w+self.crop_size]
            label = label[:, crop_anchor_h:crop_anchor_h+self.crop_size, crop_anchor_w:crop_anchor_w+self.crop_size]
        
        if self.mode == "train":
            '''Random Vertical Filp'''
            if torch.rand(()).item()>0.5:
                sample = torch.flip(sample, [1])
                label = torch.flip(label, [1])
                
            '''Random Horizental Filp'''
            if torch.rand(()).item()>0.5:
                sample = torch.flip(sample, [2])
                label = torch.flip(label, [2])
        
        ''' Normalize'''
        sample = self.normalize(sample)
        return sample, label


# Load datasets
def get_tem_Data(data_root:str=PATH, file_root: str=PATH, crop_size:int=256, train_bs:int=4, num_workers: int=0, is_syn:bool=False):

    ds_train = ImageLoader(data_root, file_root, mode="train", crop_size=crop_size, is_syn=is_syn)
    ds_test = ImageLoader(data_root, file_root, mode="test", crop_size=crop_size)
    
    # print('training data: {}, test data: {}'.format(len(ds_train), len(ds_test)))

    train_loader = DataLoader(ds_train, batch_size=train_bs, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=num_workers)
    
    input_dim, output_dim = 1, 1
    
    return train_loader, test_loader, input_dim, output_dim

if __name__ == '__main__':

    train_loader, test_loader, input_dim, output_dim = get_tem_Data()
    for inputs, targets in train_loader:
        pass
    for inputs, targets in test_loader:
        pass