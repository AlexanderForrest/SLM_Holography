#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:18:53 2023

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
from ImageStack import SLMImageHolographyStacks

import time

N_modes = 57
n_core = 1.00001
n_cladding = 1

diameter = 17.4 * u.mm
total_pixels = 1024
wavelength = 1.55 * u.micron
f_len = 400 * u.mm

entrance_diam = 5 * u.mm  # orig = 6.25mm

y_centering = -512 + 553
x_centering = -512 + 378

x_ap_shift = 0 * u.mm
y_ap_shift = -0 * u.mm

phaseimperfection = 0.1

Test_LP = SLM_DPixel(x_pixels=1024, 
                          y_pixels=1024, 
                          x_dim=17.4 * u.mm, 
                          y_dim=17.4 * u.mm, 
                          wavelength=1.55 * u.micron, 
                          e_diam = 7 * u.mm,
                          focal_length = 400 * u.mm, 
                          radian_shift = 2 * np.pi, 
                          only_in_e_diam=True, 
                          pix_per_super=2
                          )

Test_LP.LPModeEncoding(N_modes = N_modes, 
                        el = 1, 
                        m = 1, 
                        n_core = n_core, 
                        n_cladding = n_cladding, 
                        make_odd = False, 
                        oversample = 4,
                        oversize = 8)

Test_LP.AddPadding(1, 0)

#Test_LP.ImageShift(x_centering, y_centering, shift_super_pixel_array=False)

transmission = np.ones((total_pixels, total_pixels))

Final_SLM = Test_LP.DoublePixelConvert()
Final_SLM_scaled = Final_SLM / (2*np.pi + phaseimperfection) * wavelength.to(u.m).value

#Final_SLM_scaled = np.ones((total_pixels,total_pixels))

plt.pause(0.0001)

plt.figure()
plt.imshow(Final_SLM_scaled)
plt.colorbar()
plt.show()
plt.pause(0.0001)

frsys = poppy.FresnelOpticalSystem(name='Test', pupil_diameter=1 * diameter, beam_ratio = 0.5, npix = total_pixels)             # Creating the system
frwf = poppy.FresnelWavefront(beam_radius = diameter/2, oversample = 2, wavelength = wavelength, npix = total_pixels)

"""
This section creates all of the optics that will be used in the system
"""
lens1 = poppy.QuadraticLens(f_lens = f_len, name = "Lens 1")
apature1 = poppy.CircularAperture(radius = (entrance_diam.to(u.m).value / 2), pad_factor = 1.5, name = "Lens Apature 1")

SLM = poppy.ArrayOpticalElement(transmission=transmission, opd=Final_SLM_scaled, name='SLM transformation', pixelscale=(17.40 / total_pixels) * u.mm/u.pixel)#, opd = wavelength.to(u.m)/(2 * np.pi))
tilt = poppy.TiltOpticalPathDifference(name = 'SLM_tilt', tilt_angle= 1 * u.degree, rotation=90)
#tiltedSLM = poppy.TipTiltStage(SLM, radius = 17.4 * u.mm)
#tiltedSLM.set_tip_tilt(tip = 10 * u.arcsec, tilt = 0 * u.arcsec)

apature1.shift_x = x_ap_shift.to(u.m).value
apature1.shift_y = y_ap_shift.to(u.m).value

frsys.add_optic(apature1)#, shift_x = x_centering/1024 * 0.0174 * u.m, shift_y = -y_centering/1024 * 0.0174 * u.m))
#frsys.add_optic(tilt, distance = 0 * u.m)
frsys.add_optic(SLM, distance = f_len)

frsys.add_optic(lens1, distance = f_len)
frsys.add_detector(pixelscale= 10 * u.micron / u.pixel, fov_pixels = total_pixels * u.pixel, distance = f_len)

final, inter = frsys.propagate(frwf, return_intermediates=True)

plt.figure()
final.display(what = 'both', vmax = 5, colorbar = True)#, vmin = 1e-2, scale = 'log')

