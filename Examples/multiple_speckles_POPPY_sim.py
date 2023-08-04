#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 10:21:07 2023

@author: forrest
"""

import matplotlib.pyplot as plt
import numpy as np
import poppy
from poppy.poppy_core import PlaneType
import astropy.units as u
import astropy.io.fits as fits
#from lanternfiber import lanternfiber
from skimage.transform import resize, rescale

import os
import sys

from SLM_encoding_program import SLM_DPixel

pixels = 1024

dim_slm = 17.4 * u.mm
total_pixels = pixels * u.pixel
wavelength = 1.55 * u.micron
f_len = 400 * u.mm
diameter = 10 * u.mm

apature_to_SLM = 1
detector_scale = dim_slm/(15*total_pixels)

distance_from_center = [0.000000001, 10, 10, 14, 26]
amplitude = [1/5, 1/5, 1/5, 1/5, 1/5]
rotation = [0, 0, 180, 45, 90]

summed_speckel_pupil = np.zeros((pixels,pixels), dtype = 'complex')

for i in range(len(distance_from_center)):
    Y, X = np.mgrid[-pixels / 2:pixels / 2, -pixels / 2:pixels / 2]
    
    period = pixels / distance_from_center[i]
    Xr = np.cos(rotation[i] / 180 * np.pi) * X + np.sin(rotation[i] / 180 * np.pi) * Y
    im_c = amplitude[i] * np.exp(1j * 2 * np.pi / period * Xr)
    
    summed_speckel_pupil += im_c
    
transmission_array = np.abs(summed_speckel_pupil)
opd_array = np.angle(summed_speckel_pupil) / (2 * np.pi) * wavelength.to(u.m).value


frsys = poppy.FresnelOpticalSystem(name='Test', pupil_diameter=1 * dim_slm, beam_ratio = 2, npix = 1*total_pixels)             # Creating the system
frwf = poppy.FresnelWavefront(beam_radius = 1*dim_slm /2, oversample = 4, wavelength = wavelength, npix = 1*int(total_pixels.value))

"""
This section creates all of the optics that will be used in the system
"""
lens1 = poppy.QuadraticLens(f_lens = f_len, name = "Lens 1")
apature1 = poppy.CircularAperture(radius = (diameter.to(u.m).value / 2), name = "Lens Apature 1")
lens2 = poppy.QuadraticLens(f_lens = f_len, name = "Lens 2")
apature2 = poppy.CircularAperture(radius = (diameter.to(u.m).value / 2), name = "Lens Apature 2")
lens3 = poppy.QuadraticLens(f_lens = f_len, name = "Final Lens")
spatial_filter_empty = poppy.ScalarTransmission(name = 'Empty Focal Plane Placeholder')
SLM = poppy.ArrayOpticalElement(transmission=transmission_array, opd=opd_array, name='SLM transformation', pixelscale=(17.40 / (pixels)) * u.mm/u.pixel)

circle = poppy.CircularAperture(radius = 1 * u.m, name = "Lens Apature 1")

frsys.add_optic(poppy.CircularAperture(radius=diameter.to(u.m).value / 2, name = "Initial apature"))
frsys.add_optic(SLM, distance = apature_to_SLM * f_len)
frsys.add_optic(lens1, distance = 0 * u.mm)

frsys.add_detector(pixelscale = detector_scale, fov_pixels = 4000, distance = f_len)

compl, inter = frsys.propagate(frwf, return_intermediates=True)

plt.figure(3)
compl.display(what = "both", vmax = 1)
