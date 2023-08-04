#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:17:08 2023

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

pixels = 1024 # Number of Pixels for SLM

"""
This section is responsible for determining all of the variables that should be modified
"""

scale = 1
diameter = scale * 10.0 * u.mm        # General diameter for the system
f_len = scale * 10 * u.mm            # Focal Length for the two Lenses
wavelength = 1.55 * u.micron              # wavelength of light

dim_slm = 17.40 * u.mm
total_pixels = pixels * u.pixel

e_diam_pixels = int(diameter / dim_slm * total_pixels.value) + 1

sp_r_high = scale * 50 * f_len * np.tan(np.arcsin(wavelength / diameter))
sp_r_low = scale * 2.5 * u.mm

#print('The radius of the spatial filter is: ' + str(sp_r_high))

add_high = False
add_low = False

add_lens_aperture_1 = False
add_lens_aperture_2 = False

# Distances between objects, given in number of focal lengths
apature_to_SLM = 0
SLM_to_lens1 = 1
lens2_to_detector = 1

# This section controls the SLM settings

gaussian_ampl =   False
flat_ampl =       False
random_ampl =     False
interprand_ampl = False
focal_ampl =      False

random_phase =     False
interprand_phase = False
zernike_phase =    False
flat_phase =       False

interprand_both = False

LP_MMF_encoding = False

single_pair_spots =   False
multiple_pair_spots = True

# Single spot variables

central = 1
left = 1
right = 1
spacing = 20
rotation = 0

# Multiple spot variables

power_mult =    [1, 1, 1, 1, 1]
spacing_mult =  [0, 1 , 2, 3, 4]
rotation_mult = [0, 5, 90, -90, 13]

#LP modes variables

N_modes = 54
l = 2
m = 2
n_core = 1.44
n_cladding = 1.4345 # 1.4345
make_odd = False

detector_scale = dim_slm/(15*4*total_pixels)

foldername = "/Dump"
SLM_flat_ampl = 0.5
SLM_flat_phase = 1

# Gaussian amplitude variables

a = 1
b = (diameter / dim_slm)* pixels/4 - 0.5
c = 0.5 * pixels

# List of Zernike terms (Noll index) and the scaling factors applied to each

noll_index = [4]
zern_scale = [1]
zs_factor = 1

zs_sum = sum(zern_scale) / zs_factor

zern_scaling = [i / zs_sum for i in zern_scale]

# Interpolated Random Number of Points

num_ampl = 100
num_phase = 100
num_both = 100

neighbors = 50

#This section creates and writes what information should be encoded into the focal plane

# will look into making this a class/object

focal_pixels = pixels

fa_scale = np.arcsin((1.22 * wavelength.to(u.m)) / dim_slm.to(u.m))
focalArray = np.zeros((focal_pixels, focal_pixels), dtype = 'cfloat')
"""
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

#makes a circle
focalArray[r<50] += 1 * np.exp(1j * 0 * np.pi)

# makes a square
#focalArray[low:high, low:high] += 1

# makes gaussians
#for i in range(focal_pixels):
#    for j in range(focal_pixels):
#            focalArray[i,j] += 1*np.exp((-(i - bf)**2/cf) - ((j - bf)**2/cf)) * np.exp(1j * exp1phase)
#            focalArray[i,j] += 1*np.exp((-(i - bf)**2/cf) - ((j - bf - 60)**2/cf)) * np.exp(1j * exp2phase)
"""

"""
This section is responsible for setting up the SLM
"""

#SLM = SLM_information(pixels, pixels)
D_pixel = SLM_DPixel(x_pixels = pixels,
                     y_pixels = pixels,
                     x_dim = dim_slm,
                     y_dim = dim_slm,
                     wavelength = wavelength,
                     e_diam_pixels = e_diam_pixels,
                     focal_length = f_len,
                     radian_shift = 2*np.pi,
                     only_in_e_diam = True,
                     pix_per_super = 4,
                     less_than_2pi = False)      # Creates a double pixel object

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
    D_pixel.LPModeEncoding(N_modes, l, m, n_core, n_cladding, make_odd)

if random_phase == True:
    D_pixel.RandomPhase()
if interprand_phase == True:
    D_pixel.InterpolateRandomPhase(num_phase, neighbors)
if zernike_phase == True:
    D_pixel.ZernikeTerms(noll_index, zern_scaling)
if flat_phase == True:
    D_pixel.FlatPhase(SLM_flat_phase)
if single_pair_spots:
    D_pixel.focal_spot(central, left, right, spacing, rotation)
if multiple_pair_spots:
    D_pixel.focal_spots_multiple(power_mult, spacing_mult, rotation_mult)

opd = D_pixel.DoublePixelConvert()


transmission = np.full((pixels, pixels), 1)
Transformed_opd = opd * wavelength.to(u.m) / (2 * np.pi)

Transformed_opd.to(u.m)

opd_scale = 1

# Increasing the scale of the SLM

Trans_scaled_opd = np.kron(Transformed_opd.value, np.ones((opd_scale,opd_scale)))
scaled_transmission = np.kron(transmission, np.ones((opd_scale, opd_scale)))
        

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
    
frsys.add_optic(poppy.ScalarTransmission(planetype=PlaneType.image, name='Final Pupil Detector Plane'), distance = 1 * f_len)
nameList.append("Pupil Plane Detector")

frsys.add_optic(lens3, distance = 0 * u.m)
nameList.append("Final Focus Lens")
frsys.add_detector(pixelscale = detector_scale, fov_pixels = 4000, distance = f_len)
nameList.append("Final Focal Plane Detector")

print("Fresnel Optical System {} wavelength of light".format(wavelength))

#psf, waves = frsys.calc_psf(wavelength = wavelength, display_intermediates=True, return_intermediates=True)
compl, inter = frsys.propagate(frwf, return_intermediates=True)

#print(compl)
#print(inter)

plt.figure(10)
plt.title('SLM Desired Amplitude')
plt.imshow(D_pixel.SLM_ampl)
#plt.savefig('SLMDesiredAmpl.png', dpi = 800)
plt.show()

plt.figure(11)
plt.title('SLM Desired Phase')
plt.imshow(D_pixel.SLM_phase)
#plt.savefig('SLMDesiredPhase.png', dpi = 800)
plt.show()

plt.figure(1)
plt.title('The SLM Pattern')
plt.imshow(D_pixel.SLM_encoded, cmap = 'jet')
plt.colorbar()
#plt.savefig('WhatsEncodedFullSim.png', dpi = 800)
plt.show()
plt.pause(0.001)
"""
plt.figure(2)
plt.imshow(D_pixel.Amplitude)
plt.show()
plt.pause(0.001)
"""
plt.figure(3)
compl.display(what = "both", vmax = 5)

plt.figure(4, figsize = (20,14))
inter[-3].display(what = 'both', vmax_wfe = np.pi, colorbar = True, scale = 'log', use_angular_coordinates=False, imagecrop = 0.02)
#plt.savefig('Virtual_Pupil_Plane.png', dpi = 400)

PL_inject_ampl = compl.amplitude
PL_inject_phase = compl.phase
PL_inject_compl = PL_inject_ampl * np.exp(1j * PL_inject_phase)
PL_inject_scale = detector_scale

print(PL_inject_scale)

lantern_r = D_pixel.a * u.m
lantern_r = lantern_r.to(u.micron)

max_r = 2 # Maximum radius to calculate mode field, where r=1 is the core diameter
npix = 200 # Half-width of mode field calculation in pixels
show_plots = True

inp_pix_scale = PL_inject_scale.to(u.micron / u.pixel) / (lantern_r * 2 / npix) * u.pixel
print(inp_pix_scale)
plot_modefields = True
save_image_sequences = False
imseq_out_dir = './imseq/'


"""
### Make the fiber and modes
f = lanternfiber(n_core, n_cladding, lantern_r.value, wavelength.value)
f.find_fiber_modes()
f.make_fiber_modes(npix=npix, show_plots=False, max_r=max_r)
modes_to_measure = np.arange(f.nmodes)
print(f.modelabels)
# Plot all mode fields
if plot_modefields:
    plt.figure(4)
    plt.clf()
    nplots = len(f.allmodefields_rsoftorder) #7
    zlim = 0.03
    for k in range(nplots):
        plt.subplot(6,6,k+1)
        sz = f.max_r * f.core_radius
        plt.imshow(f.allmodefields_rsoftorder[k], extent=(-sz, sz, -sz, sz), cmap='bwr',
                   vmin=-zlim, vmax=zlim)
        plt.xlabel('Position ($\mu$m)')
        plt.ylabel('Position ($\mu$m)')
        core_circle = plt.Circle((0,0), f.core_radius, color='k', fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_patch(core_circle)
        plt.title(k)
    # plt.tight_layout()

# Get input fields
#input_cube = np.load(datapath+input_filename)

x,y = np.indices((int(npix/inp_pix_scale*2), int(npix/inp_pix_scale*2)))

inject_shape = np.shape(PL_inject_compl)
print(inject_shape)
#input_cube = np.zeros((2,inject_shape[0], inject_shape[1]))

input_cube = PL_inject_compl #[0,:,:]

#input_cube[1,:,:] = PL_inject_compl
#n_flds = input_cube.shape[0]
#n_flds = 1 ## For testing - just show first one
k = 0
#for k in range(n_flds):
orig_field = input_cube#[k,:,:]
resized_field_real = rescale(orig_field.real, inp_pix_scale)
resized_field_imag = rescale(orig_field.imag, inp_pix_scale)
plt.show()
resized_field = resized_field_real + resized_field_imag*1j

input_field = resized_field
cnt = input_field.shape[1]//2
input_field = input_field[cnt-f.npix:cnt+f.npix, cnt-f.npix:cnt+f.npix]


f.input_field = input_field
f.plot_injection_field(f.input_field, show_colorbar=False, logI=True, vmin=-3, fignum = 5)
plt.pause(0.001)

if save_image_sequences:
    fname = imseq_out_dir + 'injplot_%.3d' % k + '.png'
    plt.savefig(fname, bbox_inches='tight', dpi=200)

coupling, mode_coupling, mode_coupling_complex = f.calc_injection_multi(mode_field_numbers=modes_to_measure,
                                                 verbose=True, show_plots=True, fignum=6, complex=True, ylim=0.3)
### The complex LP mode coefficients are in mode_coupling_complex.

if save_image_sequences:
    fname = imseq_out_dir + 'modeplot_%.3d' % k + '.png'
    plt.savefig(fname, bbox_inches='tight', dpi=200)

plt.pause(0.5)
"""
