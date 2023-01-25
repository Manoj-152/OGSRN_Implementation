import os
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml

from torch.utils.data import Dataset, DataLoader
import torch
torch.manual_seed(10)
import random
random.seed(10)

class SAR_optic_dataset(Dataset):
    def __init__(self, cfg, tensor_transform = True):
        self.tensor_transform = tensor_transform
        self.scale_ratio = cfg['SCALE_RATIO']
        root_dir = cfg['DATASET_PATH']
        self.hor_flip_prob = 1. - cfg['DATASET']['HORIZONTAL_FLIP_PROB']
        self.ver_flip_prob = 1. - cfg['DATASET']['VERTICAL_FLIP_PROB']

        self.optic_data_paths = sorted(glob(os.path.join(root_dir, 'Optic_Images', '*.npy')))
        self.sar_data_paths = sorted(glob(os.path.join(root_dir, 'SAR_Images', '*.npy')))
        
        self.data_pairs = []
        for optic_path in self.optic_data_paths:
            optic_name = os.path.basename(optic_path)
            sar_path = os.path.join(root_dir, 'SAR_Images', optic_name)
            self.data_pairs.append((optic_path, sar_path))

    def __getitem__ (self, index):
        optic_path, sar_path = self.data_pairs[index]
        optic_img = np.load(optic_path)
        sar_img_hr = np.load(sar_path)

        width = int(sar_img_hr.shape[1] / self.scale_ratio)
        height = int(sar_img_hr.shape[0] / self.scale_ratio)
        dim = (width, height)
        sar_img_lr = cv2.resize(sar_img_hr, dim, interpolation=cv2.INTER_CUBIC)

        if self.tensor_transform == True:
            optic_img = torch.Tensor(optic_img)         
            sar_img_hr = torch.Tensor(sar_img_hr)
            sar_img_lr = torch.Tensor(sar_img_lr)
            
            if random.random() >= self.ver_flip_prob:                    # Flipping along the first dimension
                optic_img = torch.flip(optic_img, [0])
                sar_img_hr = torch.flip(sar_img_hr, [0])
                sar_img_lr = torch.flip(sar_img_lr, [0])
            if random.random() >= self.hor_flip_prob:                    # Flipping along the second dimension
                optic_img = torch.flip(optic_img, [1])
                sar_img_hr = torch.flip(sar_img_hr, [1])
                sar_img_lr = torch.flip(sar_img_lr, [1])
            
            optic_img = optic_img.unsqueeze(dim=0)
            sar_img_hr = sar_img_hr.unsqueeze(dim=0)
            sar_img_lr = sar_img_lr.unsqueeze(dim=0)
        
        return optic_img, sar_img_hr, sar_img_lr

    def __len__ (self):
        return len(self.data_pairs)


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    dataset = SAR_optic_dataset(cfg, tensor_transform=True)
    trainloader = DataLoader(dataset, batch_size=1, shuffle=False)
    x = iter(trainloader)
    o,s_hr,s_lr = x.next()
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(o[0].permute(1,2,0).squeeze().detach(), "gray")
    ax[1].imshow(s_hr[0].permute(1,2,0).squeeze().detach(), "gray")
    ax[2].imshow(s_lr[0].permute(1,2,0).squeeze().detach(), "gray")
    #plt.savefig('trial.png',dpi=150)
    plt.show()