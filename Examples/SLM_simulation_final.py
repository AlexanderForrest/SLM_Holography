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
#from SLM_encoding_program import SLM_information

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from SLM_encoding_program import SLM_DPixel


pixels = 1024 # Number of Pixels for SLM

"""
This section is responsible for determining all of the variables that should be modified
"""

scale = 1
diameter = scale * 10.0 * u.mm        # General diameter for the system
f_len = scale * 100 * u.mm            # Focal Length for the two Lenses
wavelength = 1.55 * u.micron              # wavelength of light

dim_slm = 17.40 * u.mm
radian_shift = 2 * np.pi
pix_per_super = 10

total_pixels = pixels * u.pixel

sp_r_high = scale * 250 * f_len * np.tan(np.arcsin(wavelength / diameter))
sp_r_low = scale * 2.5 * u.mm

print(sp_r_high)

add_high = True
add_low = False

add_lens_aperture_1 = False
add_lens_aperture_2 = False

# Distances between objects, given in number of focal lengths
apature_to_SLM = 0
SLM_to_lens1 = 1
lens2_to_detector = 1

# This section controls the SLM settings

gaussian_ampl = True
flat_ampl = True
random_ampl = True
interprand_ampl = False
focal_ampl = True

random_phase = True
interprand_phase = False
zernike_phase = True
flat_phase = True

interprand_both = False

LP_MMF_encoding = True
N_modes = 54
l = 3
m = 2
n_core = 1.44
n_cladding = 1.4345
make_odd = False

foldername = "/Dump"
SLM_flat_ampl = 1
SLM_flat_phase = 1

# Gaussian amplitude variables

a = 1
b = pixels/4 - 0.5
c = 4 * pixels

# List of Zernike terms (Noll index) and the scaling factors applied to each

noll_index = [7]
zern_scale = [4]
zs_factor = 4

zs_sum = sum(zern_scale) / zs_factor

zern_scaling = [i / zs_sum for i in zern_scale]

# Interpolated Random Number of Points

num_ampl = 100
num_phase = 100
num_both = 100

neighbors = 50

#This section creates and writes what information should be encoded into the focal plane
"""
pixels_focal = pixels * diameter / dim_slm / 2
pixels_focal = round(pixels_focal.value) + 1
"""

focal_pixels = 5*pixels

fa_scale = dim_slm/(8*pixels)
focalArray = np.zeros((focal_pixels, focal_pixels), dtype = 'cfloat')

cf = 0.1 * focal_pixels
bf = focal_pixels/2 - 0.5 - 30
square = 10

exp1phase = 0.5 * np.pi
exp2phase = 1 * np.pi

square_sides = 50
low = focal_pixels//4  - square_sides
high = focal_pixels//4 + square_sides

y,x = np.indices((focal_pixels,focal_pixels))

r = np.sqrt((x- focal_pixels/2)**2 + (y- focal_pixels/2)**2)

focalArray[r<=50] += 1 * np.exp(1j * 0 * np.pi)

#focalArray[low:high, low:high] += 1

#for i in range(focal_pixels):
#    for j in range(focal_pixels):
#            focalArray[i,j] += 1*np.exp((-(i - bf)**2/cf) - ((j - bf)**2/cf)) * np.exp(1j * exp1phase)
#            focalArray[i,j] += 1*np.exp((-(i - bf)**2/cf) - ((j - bf - 60)**2/cf)) * np.exp(1j * exp2phase)


"""
This section is responsible for setting up the SLM
"""

D_pixel = SLM_DPixel(x_pixels = pixels,
                     y_pixels = pixels,
                     x_dim = dim_slm,
                     y_dim = dim_slm,
                     wavelength = wavelength,
                     e_diam_pixels = int(diameter.to(u.m).value / dim_slm.to(u.m).value * total_pixels.value) + 1,
                     focal_length= f_len,
                     radian_shift = radian_shift,
                     only_in_e_diam = True,
                     pix_per_super = pix_per_super,
                     less_than_2pi = False)       # Creates a double pixel object

# Applies a gaussian amplitude to the double pixel methood

if interprand_both == True:
    D_pixel.InterpolateRandomBoth(num_both, neighbors)

if gaussian_ampl == True:
    D_pixel.GaussianAmpl(a, b, b, c, c)
if flat_ampl == True:
    D_pixel.GivenAmpl(SLM_flat_ampl)                                               # Applies a fixed amplitude adjustment to the double pixel method
if random_ampl == True:
    D_pixel.RandomAmpl()
if interprand_ampl == True:
    D_pixel.InterpolateRandomAmpl(num_ampl, neighbors)
if focal_ampl == True:
    D_pixel.FocalPlaneImage(focalArray, fa_scale, fa_scale)
if LP_MMF_encoding == True:
    D_pixel.LPModeEncoding(N_modes, l, m, n_core, n_cladding, wavelength, make_odd, oversample = 4)

if random_phase == True:
    D_pixel.RandomPhase()
if interprand_phase ==True:
    D_pixel.InterpolateRandomPhase(num_phase, neighbors)
if zernike_phase == True:
    D_pixel.ZernikeTerms(noll_index, zern_scaling)
if flat_phase == True:
    D_pixel.FlatPhase(SLM_flat_phase)


opd = D_pixel.DoublePixelConvert()


transmission = np.full((pixels, pixels), 1)
Transformed_opd = opd * wavelength.to(u.m) / (2 * np.pi)

Transformed_opd.to(u.m)

opd_scale = 1

# Increasing the scale of the SLM

Trans_scaled_opd = np.kron(Transformed_opd.value, np.ones((opd_scale,opd_scale)))
scaled_transmission = np.kron(transmission, np.ones((opd_scale, opd_scale)))
        

# SLM plots

f1 = plt.figure()
plt.imshow(Trans_scaled_opd, cmap='bwr', interpolation='nearest')
plt.title('SLM Encoded Information')
plt.colorbar()
plt.show()
plt.pause(0.001)
"""
f1 = plt.figure()
plt.imshow(D_pixel.Amplitude, cmap='bwr', interpolation='nearest')
plt.title('SLM Encoded Information')
plt.colorbar()
plt.show()
plt.pause(0.001)
"""
if focal_ampl == True:
    f2 = plt.figure()
    plt.imshow(D_pixel.transformPhase, cmap='twilight', interpolation='nearest')
    plt.title('SLM Phase')
    plt.colorbar()
    plt.show()
    plt.pause(0.001)
    
    f2 = plt.figure()
    plt.imshow(D_pixel.transformAmpl, cmap='gist_heat', interpolation='nearest')
    plt.title('SLM Amplitude')
    plt.colorbar()
    plt.show()
    plt.pause(0.001)
"""
"""
r_slm = (diameter.value/2) / (dim_slm.value * scale) * pixels
 
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
spatial_filter_low = poppy.SecondaryObscuration(secondary_radius=sp_r_low, n_supports=0, name = "Spatial Filter (Low)")  # Low Pass Spatial Filter
spatial_filter_high = poppy.CircularAperture(radius=sp_r_high.to(u.m).value, name = "Spatial Filter (High)")
spatial_filter_empty = poppy.ScalarTransmission(name = 'Empty Focal Plane Placeholder')
SLM = poppy.ArrayOpticalElement(transmission=scaled_transmission, opd=Trans_scaled_opd, name='SLM transformation', pixelscale=(17.40 * scale / (opd_scale * pixels)) * u.mm/u.pixel)

circle = poppy.CircularAperture(radius = 1 * u.m, name = "Lens Apature 1")

"""
The optics added in order:
    1. Initial apature
    2. 2nd Apature (potentially not needed)
    3. Lens 1
        potentially includes an apature restricting the diameter of the lens
    4. Spatial Filter(s)
        adds either the high frequency, low frequency, both spatial filterst, or no filter at all
    5. Lens 2
    6. Pupil Plane Detector
    7. Final Focusing Lens
    8. Focal Plane Detector (This detector sets up the scale/size of the entire optical system)
    
"""

frsys.add_optic(poppy.CircularAperture(radius=diameter.to(u.m).value / 2, name = "Initial apature"))
frsys.add_optic(SLM, distance = apature_to_SLM * f_len)

nameList = ["Setup Apature", "First Apature"]
# control for if the apature for lens 1 should be added or not

if add_lens_aperture_1 == False:
    frsys.add_optic(lens1, distance = SLM_to_lens1 * f_len)
    nameList.append("Lens 1")
else:
    frsys.add_optic(apature1, distance= SLM_to_lens1 * f_len)
    nameList.append("Apature for Lens 1")
    frsys.add_optic(lens1, distance = 0 * u.mm)
    nameList.append("Lens 1")

# control for either the high pass filter, low pass filter, both, or neither

if add_high == True and add_low != True:
    frsys.add_optic(spatial_filter_high, distance = 1 * f_len)
    nameList.append("Regular Spatial Filter")
    
elif add_low == True and add_high != True:
    frsys.add_optic(spatial_filter_low, distance = 1 * f_len)
    nameList.append("Low Spatial Filter")

elif add_low == True and add_high == True:
    frsys.add_optic(spatial_filter_low, distance = 1 * f_len)
    nameList.append("Low Spatial Filter")
    frsys.add_optic(spatial_filter_high, distance = 0 * f_len)
    nameList.append("Regular Spatial Filter")
    
else:
    #frsys.add_detector(pixelscale = 2 * u.micron / u.pixel, fov_pixels = 8000, distance = f_len)
    #frsys.add_optic(circle, distance = 1 * f_len)
    frsys.add_optic(spatial_filter_empty, distance = 1 * f_len)
    nameList.append("Empty Spatial Filter")

#control of if the apature for lens 2 should be added or not

if add_lens_aperture_2 == True:
    frsys.add_optic(apature2, distance= 1 * f_len)
    nameList.append("Apature for Lens 2")
    frsys.add_optic(lens2, distance = 0 * u.mm)
    nameList.append("Lens 2")
else:
    frsys.add_optic(lens2, distance = 1 * f_len)
    nameList.append("Lens 2")
    
frsys.add_optic(poppy.ScalarTransmission(planetype=PlaneType.image, name='Final Detector Plane'), distance = lens2_to_detector * f_len)
nameList.append("Pupil Plane Detector")

frsys.add_optic(lens3, distance = 0 * u.m)
nameList.append("Final Focus Lens")
frsys.add_detector(pixelscale = 2 * u.micron / u.pixel, fov_pixels = 8000, distance = f_len)
nameList.append("Final Focal Plane Detector")

#plt.figure(figsize=(18, 18))    # plots the system
print("Fresnel Optical System {} wavelength of light".format(wavelength))

#psf, waves = frsys.calc_psf(wavelength = wavelength, display_intermediates=True, return_intermediates=True)
compl, inter = frsys.propagate(frwf, return_intermediates=True)

print(compl)
print(inter)

# Plots the ampl, phase, and intensity of all of the optics onto the same figure

"""

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

"""
plt.show()
plt.pause(0.001)

filepath = "/home/forrest/Pictures/SLM_Images"


fig1 = plt.figure()
plt.title('Phase at ' + nameList[1])
plt.imshow(inter[1].phase, cmap='twilight')
plt.colorbar()
plt.savefig(fname = filepath + foldername + "/SLMPhase.png", dpi = 1000, format = "png")
plt.show()
plt.pause(0.001)


fig5 = plt.figure()
plt.title('Amplitude at ' + nameList[-3])
plt.imshow(inter[-3].amplitude, cmap='gist_heat', vmin = 0, vmax = 1)
plt.colorbar()
plt.savefig(fname = filepath + foldername + "/FinalAmpl.png", dpi = 1000, format = "png")
plt.show()
plt.pause(0.001)

fig1 = plt.figure()
plt.title('Phase at ' + nameList[-3])
plt.imshow(inter[-3].phase, cmap='twilight')
plt.colorbar()
plt.savefig(fname = filepath + foldername + "/FinalPhase.png", dpi = 1000, format = "png")
plt.show()
plt.pause(0.001)


fig4 = plt.figure()
for i in range(3, len(inter)):
    plt.subplot(1, (len(inter) - 3), i-2)
    plt.imshow(inter[i].intensity, vmin = 10**(-4), vmax = 1, cmap='gist_heat', norm = "log")
    plt.colorbar()
    plt.title(nameList[i])
    
plt.suptitle('Intensitys from SLM to Final Detector')
plt.show()
plt.pause(0.001)

"""
figfocal = plt.figure()
plt.title('Intensity in the focal plane')
if add_lens_aperture_1 == False:
    inter[3].display(what = "intensity", vmin = 10**(-4), scale = "log", colorbar = True, showpadding = True)
else:
    inter[4].display(what  = "intensity", vmin = 10**(-4), scale = "log", colorbar = True, showpadding = True)

plt.savefig(fname = filepath + foldername + "/FocalPlaneAmpl.png", dpi = 1000, format = "png")
plt.show()
plt.pause(0.001)
"""

figfocal = plt.figure()

if add_lens_aperture_1 == False:
    plt.title('Intensity at ' + nameList[3])
    plt.imshow(inter[3].intensity, cmap='gist_heat', vmin = 10**(-4), norm = "log")
else:
    plt.title('Intensity at ' + nameList[4])
    plt.imshow(inter[4].intensity, cmap='gist_heat', vmin = 10**(-4), norm = "log")

plt.colorbar()
plt.show()
plt.pause(0.001)



fig6 = plt.figure()
plt.title('Amplitude at ' + nameList[-1])
plt.imshow(inter[-1].amplitude, cmap='gist_heat', vmin = 10**(-5), norm = "log")
plt.colorbar()
plt.savefig(fname = filepath + foldername + "/FinalAmplFocalPlane.png", dpi = 1000, format = "png")
plt.show()
plt.pause(0.001)

fig1 = plt.figure()
plt.title('Phase at ' + nameList[3])
plt.imshow(inter[3].phase, cmap='twilight')
plt.colorbar()
plt.savefig(fname = filepath + foldername + "/FinalPhase.png", dpi = 1000, format = "png")
plt.show()
plt.pause(0.001)

figx = plt.figure()
inter[-1].display(what = "amplitude")
plt.show

#plt.savefig('testimage.pdf')

#frsys.describe()  # Describes the system
