from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from glob import glob
import random
random.seed(10)

class SAR_optic_dataset(Dataset):
    def __init__(self, cfg, tensor_transform = True):
        
        root_dir = cfg['DATASET_PATH']
        self.hor_flip_prob = 1. - cfg['DATASET']['HORIZONTAL_FLIP_PROB']
        self.ver_flip_prob = 1. - cfg['DATASET']['VERTICAL_FLIP_PROB']
        self.tensor_transform = tensor_transform

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
        sar_img = np.load(sar_path)
        if self.tensor_transform == True:
            optic_img = torch.Tensor(optic_img)         
            sar_img = torch.Tensor(sar_img)
            
            if random.random() >= self.ver_flip_prob:       # Flipping along the first dimension
                optic_img = torch.flip(optic_img, [0])
                sar_img = torch.flip(sar_img, [0])
            if random.random() >= self.hor_flip_prob:       # Flipping along the second dimension
                optic_img = torch.flip(optic_img, [1])
                sar_img = torch.flip(sar_img, [1])
            
            optic_img = optic_img.unsqueeze(dim=0)
            sar_img = sar_img.unsqueeze(dim=0)
        
        return optic_img, sar_img

    def __len__ (self):
        return len(self.data_pairs)
        
if __name__ == '__main__':
    
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    dataset = SAR_optic_dataset(cfg, tensor_transform=True)
    trainloader = DataLoader(dataset, batch_size=1, shuffle=False)
    x = iter(trainloader)
    o,s= x.next()
    plt.imshow(o[0].permute(1,2,0).squeeze().detach(), "gray")
    #.savefig('trial1.png')
    plt.show()
    plt.imshow(s[0].permute(1,2,0).squeeze().detach(), "gray")
    #plt.savefig('trial2.png')
    plt.show()