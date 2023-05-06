import numpy as np
import torch
import cv2

def minmax_scale(arr, min_val=0., max_val=255.):
    minimum = np.min(arr)
    maximum = np.max(arr)
    arr = ((arr - minimum)/(maximum - minimum))*(max_val - min_val) + min_val
    return arr

def inference(cfg, generator):
    
    input_sar_path = cfg['INFERENCE_SAR_PATH']
    input_sar = np.load(input_sar_path)
    cv2.imwrite(cfg['INFERENCE_SAR_PATH'][:-4]+'.png', minmax_scale(input_sar))
    input_sar = torch.Tensor(input_sar).unsqueeze(0).unsqueeze(0) # Converting to (1,1,256,256)
    input_sar = input_sar.to(cfg['DEVICE'])
    
    optic_gen, _ = generator(input_sar)
    optic_gen_img = optic_gen.squeeze().cpu().detach().numpy()
    
    cv2.imwrite(cfg['INFERENCE_SAVE_PATH'], minmax_scale(optic_gen_img))