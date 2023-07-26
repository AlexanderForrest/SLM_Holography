#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 11:16:18 2022

@author: forrest
"""
import matplotlib.pyplot as plt
import numpy as np
import poppy
from poppy.poppy_core import PlaneType
import astropy.units as u
import astropy.io.fits as fits
from SLM_encoding_program import SLM_information
from SLM_encoding_program import SLM_DPixel


pixels = 1024 # Number of Pixels for SLM

"""
This section is responsible for determining all of the variables that should be modified
"""

scale = 1
diameter = scale * 10 * u.mm        # General diameter for the system
f_len = scale * 100 * u.mm            # Focal Length for the two Lenses
wavelength = 1.55 * u.micron              # wavelength of light

total_pixels = 2 * 1024 * u.pixel

sp_r_high = scale * 1 * u.mm
sp_r_low = scale * 0.5 * u.mm

add_high = False
add_low = False

add_lens_aperture_1 = False
add_lens_aperture_2 = False

# Distances between objects, given in number of focal lengths
apature_to_SLM = 0
SLM_to_lens1 = 0
lens2_to_detector = 1

# This section controls the SLM settings

gaussian_ampl = False
flat_ampl = True
random_ampl = False
interprand_ampl = False

random_phase = False
interprand_phase = False

interprand_both = False

foldername = "/Dump"
SLM_flat_ampl = 0
num_ampl = 100
num_phase = 100
num_both = 100

neighbors = 50

"""
This section is responsible for setting up the SLM
"""

SLM = SLM_information(pixels, pixels)
D_pixel = SLM_DPixel(SLM)             # Creates a double pixel object
# Applies a gaussian amplitude to the double pixel methood

if interprand_both == True:
    D_pixel.InterpolateRandomBoth(num_both, neighbors)

if gaussian_ampl == True:
    D_pixel.GaussianAmpl(1, pixels/4 - 0.5, pixels/4 - 0.5, 4 * pixels, 4 * pixels)
if flat_ampl == True:
    D_pixel.GivenAmpl(SLM_flat_ampl)                                               # Applies a fixed amplitude adjustment to the double pixel method
if random_ampl == True:
    D_pixel.RandomAmpl()
if interprand_ampl == True:
    D_pixel.InterpolateRandomAmpl(num_ampl, neighbors)

if random_phase == True:
    D_pixel.RandomPhase()
if interprand_phase ==True:
    D_pixel.InterpolateRandomPhase(num_phase, neighbors)

opd = D_pixel.DoublePixelConvert()


transmission = np.full((pixels, pixels), 1)
Transformed_opd = opd * wavelength.to(u.m) / (2 * np.pi)

Transformed_opd.to(u.m)

opd_scale = 4

Trans_scaled_opd = np.kron(Transformed_opd.value, np.ones((opd_scale,opd_scale)))
scaled_transmission = np.kron(transmission, np.ones((opd_scale, opd_scale)))
        
f1 = plt.figure()
plt.imshow(Trans_scaled_opd, cmap='bwr', interpolation='nearest')
plt.title('SLM Encoded Information')
plt.colorbar()
plt.show()
plt.pause(0.001)

f2 = plt.figure()
plt.imshow(D_pixel.SLM_ampl, cmap='gist_heat', interpolation='nearest')
plt.title('SLM Amplitude')
plt.colorbar()
plt.show()
plt.pause(0.001)

dim_slm = 17.40 * u.mm
r_slm = (diameter.value/2) / (dim_slm.value * scale) * pixels

 
frsys = poppy.FresnelOpticalSystem(name='Test', pupil_diameter=1 * diameter, beam_ratio = 6, npix = total_pixels)             # Creating the system
frwf = poppy.FresnelWavefront(beam_radius = diameter/2, oversample = 6, wavelength = wavelength)

"""
This section creates all of the optics that will be used in the system
"""
lens1 = poppy.QuadraticLens(f_lens = f_len, name = "Lens 1")
apature1 = poppy.CircularAperture(radius = (diameter.to(u.m).value / 2), name = "Lens Apature 1")
lens2 = poppy.QuadraticLens(f_lens = f_len, name = "Lens 2")
apature2 = poppy.CircularAperture(radius = (diameter.to(u.m).value / 2), name = "Lens Apature 2")
lens3 = poppy.QuadraticLens(f_lens = f_len, name = "Final Lens")
spatial_filter_low = poppy.SecondaryObscuration(secondary_radius=sp_r_low, n_supports=0, name = "Spatial Filter (Low)")  # Low Pass Spatial Filter
spatial_filter_high = poppy.CircularAperture(radius=sp_r_high.to(u.m).value, name = "Spatial Filter (High)")
spatial_filter_empty = poppy.ScalarTransmission(name = 'Empty Focal Plane Placeholder')
SLM = poppy.ArrayOpticalElement(transmission=scaled_transmission, opd=Trans_scaled_opd, name='SLM transformation', pixelscale=(17.40 * scale / (opd_scale * pixels)) * u.mm/u.pixel)



    
frsys.add_optic(poppy.CircularAperture(radius=diameter.to(u.m).value / 2, name = "Initial apature"))
frsys.add_optic(SLM, distance = apature_to_SLM * f_len)

# control for if the apature for lens 1 should be added or not

if add_lens_aperture_1 == False:
    frsys.add_optic(lens1, distance = SLM_to_lens1 * f_len)
else:
    frsys.add_optic(apature1, distance= SLM_to_lens1 * f_len)
    frsys.add_optic(lens1, distance = 0 * u.mm)

# control for either the high pass filter, low pass filter, both, or neither

if add_high == True and add_low != True:
    frsys.add_optic(spatial_filter_high, distance = 1 * f_len)
    
if add_low == True and add_high != True:
    frsys.add_optic(spatial_filter_low, distance = 1 * f_len)
    
if add_low == True and add_high == True:
    frsys.add_optic(spatial_filter_low, distance = 1 * f_len)
    frsys.add_optic(spatial_filter_high, distance = 0 * f_len)
    
else:
    frsys.add_optic(spatial_filter_empty, distance = 1 * f_len)

#control of if the apature for lens 2 should be added or not

if add_lens_aperture_2 == True:
    frsys.add_optic(apature2, distance= 1 * f_len)
    frsys.add_optic(lens2, distance = 0 * u.mm)
else:
    frsys.add_optic(lens2, distance = 1 * f_len)
    
frsys.add_optic(poppy.ScalarTransmission(planetype=PlaneType.image, name='Final Detector Plane'), distance = lens2_to_detector * f_len)

frsys.add_optic(lens3, distance = 0 * u.m)
frsys.add_detector(pixelscale = 1 * u.micron / u.pixel, fov_pixels = 6000, distance = f_len)


#plt.figure(figsize=(18, 18))    # plots the system
print("Fresnel Optical System {} wavelength of light".format(wavelength))

#psf, waves = frsys.calc_psf(wavelength = wavelength, display_intermediates=True, return_intermediates=True)
compl, inter = frsys.propagate(frwf, return_intermediates=True)


print(compl)
print(inter)

fig, axes = plt.subplots(len(inter), 3, figsize=(10, 18))
plt.title("Phase, Amplitude and Intensity at each of the Optics")
for i in range(len(inter)):
    
    if inter[i].amplitude.max() >= 1:
        int_vmax = 1
    else:
        int_vmax = inter[i].amplitude.max()
    
    im1 = axes[i, 0].imshow(inter[i].amplitude, vmin = inter[i].amplitude.min(), vmax = inter[i].amplitude.max(), cmap='gray')
    plt.colorbar(im1, ax=axes[i, 0])

    im2 = axes[i, 1].imshow(inter[i].phase, cmap='seismic')
    plt.colorbar(im2, ax=axes[i, 1])
    
    if inter[i].intensity.max() >= 1:
        int_vmax = 1
    else:
        int_vmax = inter[i].intensity.max()
    
    im3 = axes[i, 2].imshow(inter[i].intensity, vmin = inter[i].intensity.min(), vmax = int_vmax, cmap='gist_heat')
    plt.colorbar(im3, ax=axes[i, 2])


plt.show()
plt.pause(0.001)

filepath = "/home/forrest/Pictures/SLM_Images"


fig1, ax1 = plt.subplots()
plt.title('Phase at SLM')
ax1.imshow(inter[1].phase, cmap='seismic')
plt.savefig(fname = filepath + foldername + "/SLMPhase", dpi = 1000, format = "png")
plt.show()
plt.pause(0.001)


fig5 = plt.figure()
plt.title('Amplitude at the Final Detector')
plt.imshow(inter[-3].amplitude, cmap='gist_heat', vmin = 0, vmax = 1)
plt.colorbar()
plt.savefig(fname = filepath + foldername + "/FinalAmpl", dpi = 1000, format = "png")
plt.show()
plt.pause(0.001)

fig1 = plt.figure()
plt.title('Phase at the Final Detector')
plt.imshow(inter[-3].phase, cmap='seismic')
plt.savefig(fname = filepath + foldername + "/FinalPhase", dpi = 1000, format = "png")
plt.show()
plt.pause(0.001)

name = []

fig2, ax2 = plt.subplots(1, (len(inter) - 3))
plt.title('Intensitys from SLM to Final Detector')

for i in range(3, len(inter)):
    
    color = ax2[i-3].imshow(inter[i].intensity, vmin = inter[i].intensity.min(), vmax = 1, cmap='gist_heat')
    #plt.colorbar(color, ax2[i-3])
    
plt.show()
plt.pause(0.001)

figfocal = plt.figure()
plt.title('Intensity in the focal plane')
if add_lens_aperture_1 == False:
    plt.imshow(inter[3].intensity, cmap='gist_heat', vmin = 10**(-4), norm = "log")
else:
    plt.imshow(inter[4].intensity, cmap='gist_heat', vmin = 10**(-4), norm = "log")

plt.colorbar()
plt.show()
plt.pause(0.001)

fig6 = plt.figure()
plt.title('Amplitude at the Final Focal Plane')
plt.imshow(inter[-1].amplitude, cmap='gist_heat', vmin = 10**(-4), norm = "log")
plt.colorbar()
plt.savefig(fname = filepath + foldername + "/FinalAmplPSF", dpi = 1000, format = "png")
plt.show()
plt.pause(0.001)

#plt.savefig('testimage.pdf')

#frsys.describe()  # Describes the system
