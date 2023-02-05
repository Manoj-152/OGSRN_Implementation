from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from glob import glob
import random
import cv2
from PIL import Image
from torchvision import transforms
# random.seed(10)

class TMC_optic_dataset(Dataset):
    def __init__(self, cfg, tensor_transform = True):
        
        root_dir = cfg['DATASET_PATH']
        self.hor_flip_prob = 1. - cfg['DATASET']['HORIZONTAL_FLIP_PROB']
        self.ver_flip_prob = 1. - cfg['DATASET']['VERTICAL_FLIP_PROB']
        self.tensor_transform = tensor_transform

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

        if self.tensor_transform == True:
            transform_list = []
            if random.random() >= self.ver_flip_prob:
                transform_list.append(transforms.RandomVerticalFlip(1))
            if random.random() >= self.hor_flip_prob:
                transform_list.append(transforms.RandomHorizontalFlip(1))
            transform_list.append(transforms.ToTensor())
            self.transform_img = transforms.Compose(transform_list)
            
            optic_img = self.transform_img(optic_img)
            tmc_img = self.transform_img(tmc_img)
        
        return optic_img, tmc_img

    def __len__ (self):
        return len(self.data_pairs)
        
if __name__ == '__main__':
    
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    dataset = TMC_optic_dataset(cfg, tensor_transform=True)
    trainloader = DataLoader(dataset, batch_size=1, shuffle=False)
    x = iter(trainloader)
    o,s= x.next()
    print(o[0].shape)
    plt.imshow(o[0].permute(1,2,0).squeeze().detach())
    #.savefig('trial1.png')
    plt.show()
    print(s[0].shape)
    plt.imshow(s[0].permute(1,2,0).squeeze().detach(), "gray")
    #plt.savefig('trial2.png')
    plt.show()