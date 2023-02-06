import os
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
from torchvision import transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
torch.manual_seed(10)
import random
random.seed(10)

class TMC_optic_dataset(Dataset):
    def __init__(self, cfg, tensor_transform = True):
        self.tensor_transform = tensor_transform
        self.scale_ratio = cfg['SCALE_RATIO']
        root_dir = cfg['DATASET_PATH']
        self.hor_flip_prob = 1. - cfg['DATASET']['HORIZONTAL_FLIP_PROB']
        self.ver_flip_prob = 1. - cfg['DATASET']['VERTICAL_FLIP_PROB']
        self.scale_list = [1, 2, 4]

        self.optic_data_paths = sorted(glob(os.path.join(root_dir, 'ohrc_data', '*.jpg')))
        self.tmc_data_paths = sorted(glob(os.path.join(root_dir, 'tmc_data', '*.jpg')))
        
        self.data_pairs = []
        for tmc_path in self.tmc_data_paths:
            tmc_name = os.path.basename(tmc_path)
            name_parts = tmc_name.split('_')
            name_parts[-3] = 'bot'
            optic_name = '_'.join(name_parts)
            optic_path = os.path.join(root_dir, 'ohrc_data', optic_name)
            self.data_pairs.append((optic_path, tmc_path))

    def __getitem__ (self, index):
        optic_path, tmc_path = self.data_pairs[index]
        optic_img = Image.open(optic_path).convert('RGB')
        tmc_img = Image.open(tmc_path).convert('L')

        temp_random = random.random()
        if temp_random <= 0.33:
            width1 = int(tmc_img.shape[1]/16)
            height1 = int(tmc_img.shape[0]/16)
            dim1 = (width1, height1)
            width2 = int(tmc_img.shape[1]/4)
            height2 = int(tmc_img.shape[0]/4)
            dim2 = (width2, height2)
            tmc_img_lr = cv2.resize(tmc_img, dim1, interpolation=cv2.INTER_CUBIC)
            tmc_img_hr = cv2.resize(tmc_img, dim2, interpolation=cv2.INTER_CUBIC)
            res_label = 1
        elif temp_random <= 0.67:
            width1 = int(tmc_img.shape[1]/8)
            height1 = int(tmc_img.shape[0]/8)
            dim1 = (width1, height1)
            width2 = int(tmc_img.shape[1]/2)
            height2 = int(tmc_img.shape[0]/2)
            dim2 = (width2, height2)
            tmc_img_lr = cv2.resize(tmc_img, dim1, interpolation=cv2.INTER_CUBIC)
            tmc_img_hr = cv2.resize(tmc_img, dim2, interpolation=cv2.INTER_CUBIC)
            res_label = 2
        else:
            width1 = int(tmc_img.shape[1]/4)
            height1 = int(tmc_img.shape[0]/4)
            dim1 = (width1, height1)
            width2 = int(tmc_img.shape[1]/1)
            height2 = int(tmc_img.shape[0]/1)
            dim2 = (width2, height2)
            tmc_img_lr = cv2.resize(tmc_img, dim1, interpolation=cv2.INTER_CUBIC)
            tmc_img_hr = cv2.resize(tmc_img, dim2, interpolation=cv2.INTER_CUBIC)
            res_label = 4

        if self.tensor_transform == True:
            transform_list = []
            if random.random() >= self.ver_flip_prob:
                transform_list.append(transforms.RandomVerticalFlip(1))
            if random.random() >= self.hor_flip_prob:
                transform_list.append(transforms.RandomHorizontalFlip(1))
            transform_list.append(transforms.ToTensor())
            self.transform_img = transforms.Compose(transform_list)
            
            optic_img = self.transform_img(optic_img)
            tmc_img_lr = self.transform_img(tmc_img_lr)
            tmc_img_hr = self.transform_img(tmc_img_hr)
        
        res_label = torch.Tensor(res_label)        
        return optic_img, tmc_img_hr, tmc_img_lr, res_label

    def __len__ (self):
        return len(self.data_pairs)


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    dataset = TMC_optic_dataset(cfg, tensor_transform=True)
    trainloader = DataLoader(dataset, batch_size=1, shuffle=False)
    x = iter(trainloader)
    o,s_hr,s_lr,res_label = x.next()
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(o[0].permute(1,2,0).squeeze().detach(), "gray")
    ax[1].imshow(s_hr[0].permute(1,2,0).squeeze().detach(), "gray")
    ax[2].imshow(s_lr[0].permute(1,2,0).squeeze().detach(), "gray")
    print(res_label)
    #plt.savefig('trial.png',dpi=150)
    plt.show()