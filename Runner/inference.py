import numpy as np
import torch
import cv2
from tqdm import tqdm
from glob import glob
import os

from Preprocessing.preprocess import *


def inference(cfg, srun_model):
    sar_path_dir = cfg['INFERENCE_SAR_DIR']
    os.makedirs(cfg['INFERENCE_SAVE_DIR'], exist_ok=True)
    sar_paths = sorted(glob(sar_path_dir + '/*.tif'))
    for sar_path in sar_paths:
        clipped_sar,_,_,_,_ = clipTile(sar_path)
        clipped_sar = clipped_sar [0, :, :]
        
        scaling_method = cfg['INFERENCE_SCALING']
        if scaling_method == 'histogram':
            clipped_sar = minmax_scale(clipped_sar, 0., 1.)
            clipped_sar = despeckling(clipped_sar).astype('float32')
            clipped_sar = minmax_scale(clipped_sar, 0., 1.)
            clipped_sar = histogram_scaling(clipped_sar).astype('float32')
            
        elif scaling_method == 'adaptive histogram':
            [low_limit, high_limit] = cfg['INFERENCE_ADAPTIVE_CLIP_LIMITS']
            clipped_sar = np.clip(clipped_sar, low_limit, high_limit)  # parameters to set
            clipped_sar = minmax_scale(clipped_sar, 0., 1.)
            clipped_sar = despeckling(clipped_sar).astype('float32')
            clipped_sar = minmax_scale(clipped_sar, 0., 1.)
            clipped_sar = adaptive_histogram_scaling(clipped_sar).astype('float32')
            
        else:
            raise Exception('Please enter a valid scaling method in config file')
        
        save_path = cfg['INFERENCE_SAVE_DIR'] + '/' + os.path.basename(sar_path)[:-4] + '/' + cfg['INFERENCE_SCALING']
        os.makedirs(cfg['INFERENCE_SAVE_DIR'] + '/' + os.path.basename(sar_path)[:-4], exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
        clipped_sar_img = minmax_scale(clipped_sar, 0., 255.)
        cv2.imwrite(save_path + '/despeckled_sar.png', clipped_sar_img)
        
        # Preprocessing are done here.
        # Now patches are cropped and sent to the super resolution model
        clipped_sar = minmax_scale(clipped_sar, -1., 1.)
        initial_width, initial_height = clipped_sar.shape[0], clipped_sar.shape[1]
        scale_ratio = cfg['SCALE_RATIO']
        patch_size = int(256 // scale_ratio)
        
        patches, whole_width, whole_height = crop(clipped_sar, patch_size)
        print('Total number of patches: ', len(patches))
        device = cfg['DEVICE']
        srun_model = srun_model.to(device)
        
        super_resolved = []
        for patch in tqdm(patches):
            tensor_patch = torch.Tensor(patch).unsqueeze(0).unsqueeze(0).to(device)
            out,_ = srun_model(tensor_patch)
            out_img = out.cpu().detach().numpy().squeeze()
            super_resolved.append(out_img)
            
        super_resolved_stitched = stitch_patches(super_resolved, whole_width*scale_ratio, whole_height*scale_ratio)
        super_resolved_stitched = super_resolved_stitched[:initial_width*scale_ratio, :initial_height*scale_ratio]
        # in the previous line, we crop out the padding we added earlier in crop function so that we were able to divide the whole image into patches without leaving any remains
        
        super_resolved_sar_img = minmax_scale(super_resolved_stitched,0.,255.)
        cv2.imwrite(save_path + '/super_resolved_sar.png', super_resolved_sar_img)
        print('Final size of super resolved image: ', super_resolved_stitched.shape[0], super_resolved_stitched.shape[1])