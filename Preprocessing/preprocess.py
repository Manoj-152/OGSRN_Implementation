import rasterio
import numpy as np
import bm3d

from Preprocessing.scaling import FeatureScaling

# Adding in the preprocessing functions
def clipTile(tile_path):
    img = rasterio.open(tile_path).read().astype('float32')
    img[img == 0] = np.nan
    xmin, xmax, ymin, ymax = 0, img.shape[1], 0 , img.shape[0]
    m = np.where((img.sum(axis=2) > 0).any(1))
    if m[0].shape[0] != 0:
        ymin, ymax = np.amin(m), np.amax(m) + 1
    m = np.where((img.sum(axis=2) > 0).any(0))
    if m[0].shape[0] != 0:
        xmin, xmax = np.amin(m), np.amax(m) + 1
    img = img[ymin:ymax, xmin:xmax]
    img = np.nan_to_num(img)
    return img, ymin, ymax, xmin, xmax
    
def adaptive_histogram_scaling(band):
    index = 0
    seed = 103
    obj = FeatureScaling('adaptive_hist')
    band = obj.fit(band, seed)
    return band

def histogram_scaling(band):
    index = 0
    seed = 103
    obj = FeatureScaling('histogram')
    band = obj.fit(band, seed)
    return band

def minmax_scale(arr, min_val=-1., max_val=1.):
    minimum = np.nanmin(arr)
    maximum = np.nanmax(arr)
    arr = ((arr - minimum)/(maximum - minimum))*(max_val - min_val) + min_val
    return arr

def despeckling(band, sigma_psd = 20/255, stage_arg = bm3d.BM3DStages.ALL_STAGES):
    denoised_image = bm3d.bm3d(band, sigma_psd=sigma_psd, stage_arg=stage_arg)
    return denoised_image
    
def crop(band_sar, crop_size = 256):

    new_width = int((int(band_sar.shape[0] // crop_size) + 1) * crop_size)
    new_height = int((int(band_sar.shape[1] // crop_size) + 1) * crop_size)
    new_band_sar = -1 * np.ones((new_width, new_height)).astype('float32')
    new_band_sar[:band_sar.shape[0], :band_sar.shape[1]] = band_sar

    num_sampled_0 = int(new_band_sar.shape[0] // crop_size)
    num_sampled_1 = int(new_band_sar.shape[1] // crop_size)
    itera_0 = 0
    itera_1 = 0
    patches_sar = []
    for idx0 in range(num_sampled_0):
        itera_1 = 0
        for idx1 in range(num_sampled_1):
          new_band_sar_1 = new_band_sar[itera_0 : itera_0 + crop_size, itera_1 :itera_1 + crop_size]
          patches_sar.append(new_band_sar_1)
          itera_1 = itera_1 + crop_size
        itera_0 = itera_0 + crop_size
    return patches_sar, new_width, new_height    

def stitch_patches(patches, width, height):
    patch_size = patches[0].shape[0]
    num_sampled_0 = int(width // patch_size)
    num_sampled_1 = int(height // patch_size)
    stitched_img = np.ones((width, height)).astype('float32')
    stitched_img = -1 * stitched_img

    patch_cnt = 0
    itera_0 = 0
    itera_1 = 0
    for idx0 in range(num_sampled_0):
        itera_1 = 0
        for idx1 in range(num_sampled_1):
          stitched_img[itera_0 : itera_0 + patch_size, itera_1 :itera_1 + patch_size] = patches[patch_cnt]
          itera_1 = itera_1 + patch_size
          patch_cnt += 1
        itera_0 = itera_0 + patch_size
    return stitched_img