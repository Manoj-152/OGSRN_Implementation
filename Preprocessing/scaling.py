import os
import rasterio as rs
import torch
import tensorflow as tf
import rasterio as rs
import numpy as np
from skimage import exposure


class FeatureScaling(object):
    """
      Feature scaling: normalize the range of independent variables or features of data.
	    It currently supports three types of normalization: (customised Min-Max) , (Min-Max + Gaussian) and Histogram based Normalization.
	 
    Args:
        normalization (string, default): normalization technique if not specified defaults to histogram based.
        @TODO:range(tuple, default): Define normalization range if not specified defaults to (0,1)
        @TODO: num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        param: (dictionary, optional) loading parameters
    """
    def __init__(self,
                 normalization: str = 'histogram',
                 range_l: tuple = (0,1),
                 num_workers: int = 0,
                 param: dict = {}
                 ):

      if num_workers < 0:
          raise ValueError('num_workers option should be non-negative; '
                          'use num_workers=0 to disable multiprocessing.')
      self.num_workers = num_workers
      self.normalization = normalization
      self.range = range_l
      self.param = param

      
    def _minMax(self,
                band):
      band[band != band] = 0
      min_r = band.min()
      max_r = band.max()
      x = 2.0 * (band - min_r) / (max_r - min_r) - 1.0
      return x
      
    def _histNorm(self, 
                  band):
      band[band == 0] = np.nan
      mean = np.nanmean(band)
      std =  2 * np.nanstd(band)
      band[band > mean + std] = mean + std
      band = band - (mean - std)
      band[band < 0] = np.nan
      band = band / np.nanmax(band)
      band[band != band] = 0
      return band
      
    def _adapHistNorm(self, 
                  band, clip_limit = 0.01):
      band = exposure.equalize_adapthist(band, clip_limit=clip_limit)
      return band
      
    def _minMaxGaussian(self, 
                        c, 
                        gamma, 
                        band):
      band[band == 0] = np.nan
      band = band - np.nanmin(band)
      band[band < 0] = np.nan
      band = band / np.nanmax(band)
      band[band != band] = 0
      gamma_corrected = c * np.power(band, gamma)  
      return gamma_corrected
      
    def fit(self, 
            band,
            seed = None,
            params = {}):
                
        if seed is not None:
            np.random.seed(seed)
            
        if self.normalization == "minmax":
            return self._minMax(band)
            
        elif self.normalization == "histogram":
            return self._histNorm(band)
            
        elif self.normalization == "minmaxGaussian":
            return self._minMaxGaussian(params['c'], params['gamma'], band)
        
        elif self.normalization == 'adaptive_hist':
            return self._adapHistNorm(band)

        else:
            raise ValueError('Incorrect normalization specified')