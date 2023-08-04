#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:12:27 2023

@author: forrest
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import scipy as sp
import os
import datetime
import random
import poppy
import scipy.fft as fft
import astropy.units as u
import ofiber
from skimage.transform import resize, rescale
from SLM_encoding_program import SLM_DPixel
import copy
import time

start = time.time()

pixels = 1024
dim_slm = 17.4 * u.mm
wavelength = 1.55 * u.micron
e_diam_pixels = int(6.5 / 17.4 * pixels) + 1
f_len = 400 * u.mm
pix_per_super = 2

num_runs = 10

width_2 = 35

save_dir = './temp_files/'
filename = '20230728_four_speckles_stack_TEST'

slmglobal_params = np.array([e_diam_pixels, pix_per_super, num_runs], dtype = 'float16')
all_slmim_param = np.zeros((num_runs, 3, 4), dtype = 'float32')
all_slmims = np.zeros((num_runs, e_diam_pixels,e_diam_pixels), dtype = 'float32')
all_pupims = np.zeros((num_runs, (int(e_diam_pixels/pix_per_super)+1), (int(e_diam_pixels/pix_per_super)+1)), dtype = 'complex64')
all_psfs = np.zeros((num_runs, width_2 * 2, width_2 * 2), dtype = 'complex64')

padding = int(e_diam_pixels)

for i in range(num_runs):

    power_mult =   0.20 * np.random.rand(4) + 0.05
    spacing_mult =  2.5 * np.random.rand(4)
    rotation_mult = 360 * np.random.rand(4)
    
    im_vars = np.zeros((3,4))
    im_vars[0] = power_mult
    im_vars[1] = spacing_mult
    im_vars[2] = rotation_mult
    
    all_slmim_param[i] = im_vars
    
    mask_diam = e_diam_pixels // pix_per_super + 1
    mask = np.zeros((mask_diam, mask_diam), dtype = 'bool')
    Y, X = np.mgrid[-mask_diam / 2:mask_diam / 2, -mask_diam / 2:mask_diam / 2]
    R = np.sqrt(X ** 2 + Y ** 2)
    mask[R < mask_diam//2] = True
    
    
    D_pixel = SLM_DPixel(x_pixels = pixels,
                         y_pixels = pixels,
                         x_dim = dim_slm,
                         y_dim = dim_slm,
                         wavelength = wavelength,
                         e_diam_pixels = e_diam_pixels,
                         focal_length = f_len,
                         radian_shift = 2*np.pi,
                         only_in_e_diam = True,
                         pix_per_super = pix_per_super,
                         less_than_2pi = False)
    
    D_pixel.focal_spots_multiple(power_mult, 
                                 spacing_mult, 
                                 rotation_mult)
    
    #plt.figure(1)
    #plt.imshow(np.abs(D_pixel.im_c))
    #plt.show()
    
    #plt.figure(2)
    #plt.imshow(np.angle(D_pixel.im_c), cmap = 'hsv')
    #plt.show()
    
    all_pupims[i] = D_pixel.im_c
    
    SLM_array = D_pixel.DoublePixelConvert(add_padding = False)
    
    all_slmims[i] = SLM_array
    
    #plt.figure(3)
    #plt.imshow(SLM_array)
    #plt.show()
    
    SLM_array_padded = np.pad(D_pixel.im_c * mask, padding)
    
    focal_plane_array = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(SLM_array_padded)))
    
    center = focal_plane_array.shape[0]//2
    
    focal_plane_crop = focal_plane_array[center - width_2:center + width_2, center - width_2:center + width_2]
    
    all_psfs[i] = focal_plane_crop
    
    if i%10 == 0:
        print(i)
    
    #plt.figure(4)
    #plt.imshow(np.abs(focal_plane_crop))
    #plt.show()
    
    #plt.figure(5)
    #plt.imshow(np.angle(focal_plane_crop), cmap = 'hsv')
    #plt.show()

end = time.time()

print('SLM global params:')
print(slmglobal_params.shape)
print(slmglobal_params.dtype)

print('SLM im_params:')
print(all_slmim_param.shape)
print(all_slmim_param.dtype)

print('SLM images:')
print(all_slmims.shape)
print(all_slmims.dtype)

print('pupil images:')
print(all_pupims.shape)
print(all_pupims.dtype)

print('psf images:')
print(all_psfs.shape)
print(all_psfs.dtype)

elapsed = end - start

print('Time_elapsed: ' + str(elapsed))

np.savez_compressed(save_dir + filename, 
                    slmglobal_params = slmglobal_params, 
                    all_slmim_param = all_slmim_param, 
                    all_slmims = all_slmims,
                    all_pupims = all_pupims,
                    all_psfs = all_psfs)

plt.figure(1)
plt.imshow(np.abs(D_pixel.im_c))
plt.show()

plt.figure(4)
plt.imshow(np.abs(focal_plane_crop))
plt.show()
