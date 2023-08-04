# -*- coding: utf-8 -*-
"""
This is code to try and encode the double pxel method of phase and amplitude modulation onto
a phase only SLM
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
"""
class SLM_information(object):
    
    def __init__(self, x_pixels, y_pixels):
        
        self.x_pix = x_pixels
        self.y_pix = y_pixels
        
        self.SLM_encoded = np.empty([y_pixels, x_pixels])
        
    def CreateFolder(self, filename):
        dir_here = os.getcwd()
        whats_here = os.listdir(dir_here)
        currentdate = datetime.datetime.now().strftime('%d-%m-%Y %H:%M')

        file_name_time = filename + ' ' + currentdate

        if whats_here.count(file_name_time) == 0:
            
            self.path = os.path.join(dir_here, file_name_time)
            os.mkdir(self.path)
        else:
            pass
        
    def SaveSLM(self, filename, folder_directory = None):
        self.folder_directory = folder_directory if not None else self.path
        pass
"""
class SLM_DPixel(object):
    """
    A class that can create holograms when using an ideal phase-only SLM
    
    ...
    
    Attributes (UNFINISHED SECTION)
    ----------
    pixels_x: int
        The number of double pixels in the x direction (1/2 total pixels rounded down)
    pixels_y: int
        The number of double pixels in the y direction (1/2 total pixels rounded down)
    x_dim: float      astropy.units.Quantity, convertable to meters
        The total width of the SLM in the x direction
    y_dim: float      astropy.units.Quantity, convertable to meters
        The total width of the SLM in the y direction
    focal_length: astropy.units.Quantity, convertable to meters
        (UNUSED) The focal length of the lens used to focus the wavefront when forming a PSF
    radian_shift: float
        The maximum phase shift the SLM can apply in radians
    x_pixscale: float
        The dimensions of each double pixel in the x direction, assuming equal size
    y_pixscale: float
        The dimensions of each double pixel in the x direction, assuming equal size
    e_diam: astropy.units.Quantity, convertable to meters
        The diameter of the circualr entrance pupil, or the diameter of the beam of light onto the SLM
    only_in_e_diam: bool
        If True, only encodes information into the SLM area that overlaps with the e_diam,
        and then pads at the end to recover the original SLM dimensions.
    
    SLM_ampl: ndarray
        An array to contain the desired amplitude of the wavefront, with dimensions 
        euqal to the number of double pixels and with each value being [0,1]
    SLM_phase: ndarray
        An array to contain the desired phase of the wavefront, with dimensions
        equal to the number of double pixels and each value being [0,1].
    SLM_encoded: ndarray
        The final array with the deisred holography encoded, wiht dimensions 
        equal to the number of total pixels on the SLM
    
    Methods
    -------
    DoublePixelConvert()
        Takes the requested SLM amplitude and Phase, encodes the hologram onto 
        to SLM_encoded, and returns SLM_encoded
    GaussianAmpl(a, b_x, b_y, c_x, c_y)
        Adds a gaussian amplitude to SLM_ampl
    MaxAmpl()
        Sets all values in SLM_ampl to 1
    GivenAmpl(ampl)
        Adds a given aplitude to all values in SLM_ampl
    RandomAmpl()
        Adds a random value to each position in SLM_ampl
    RandomPhase()
        Adds a random value to each position in SLM_phase
    InterpolateRandomAmpl(num_points, neighbors)
        Adds a continuous random 2D array to SLM_ampl using scipy.interpolate.RBFInterpolator
    InterpolateRandomPhase(num_points, neighbors)
        Adds a continuous random 2D array to SLM_phase using scipy.interpolate.RBFInterpolator
    InterpolateRandomBoth(num_points, neighbors)
        Adds the same continuous random 2D array to SLM_ampl and SLM_phase using scipy.interpolate.RBFInterpolator
    CustomAmpl(custom_ampl)
        Adds a given array to SLM_ampl
    CustomPhase(custom_phase)
        Adds a given array to SLM_phase
    ResetBoth()
        Sets all vlaues in SLM_ampl and SLM_phase to zero
    ImageShift(x_shift, y_shift)
        rolls SLM_ampl and SLM_phase by a given number of double pixels to shift the image left or right
    ZernikeTerms(j_list, j_scaling, D = 1, SLM_width = 1)
        Adds zernike terms to SLM_phase only within the beam diameter.  
        Can add multiple zernike terms each with different relative ampltiudes
    FlatPhase()
        Adds a given phase in terms of [0,1] to SLM_phase
    
    """
    
    def __init__(self, x_pixels, y_pixels, x_dim, y_dim, wavelength, e_diam_pixels, focal_length = 1 * u.mm, radian_shift = 4*np.pi, only_in_e_diam = True, pix_per_super = 2, less_than_2pi = False):
        """
        

        Parameters
        ----------
        x_pixels : int
            The number of pixels in the x direction of the SLM.
        y_pixels : int
            The number of pixels in the y direction of the SLM.
        x_dim : astropy.units.Quantity, convertable to meters
            The total width(x-direction) of the SLM.
        y_dim : astropy.units.Quantity, convertable to meters
            The total height(y-direction) of the SLM.
        wavelength : astropy.units.Quantity, convertable to meters
            The wavelength of light being modified by the SLM.
        e_diam : astropy.units.Quantity, convertable to meters
            The diamater of the circular entrance pupil of light, assumed to be a top hat.
        focal_length : astropy.units.Quantity, convertable to meters, optional
            The focal length of the final lens. The default is 1 * u.mm.
        radian_shift : float, optional
            The maximum phase shift that the SLM can produce, from [0,radian_shoft]. The default is 4*np.pi.
        only_in_e_diam : bool, optional
            If true, the SLM array is truncated to only be the same width and height as the e_diam. The default is True.
        pix_per_super : int, multiple of 2, optional
            The number of pixels per super pixel.  original is 2, or a 2x2 superpixel
            
        Returns
        -------
        None.

        """
        
        if pix_per_super < 2 and pix_per_super % 2 != 0:
            raise Exception("pix per super needs to be greater than 2 and a multiple of 2")
        
        self.total_x_pixels = x_pixels
        self.total_y_pixels = y_pixels
        
        self.pixels_x_orig = x_pixels//pix_per_super + 1
        self.pixels_y_orig = y_pixels//pix_per_super + 1
        self.x_dim_orig = x_dim.to(u.m).value
        self.y_dim_orig = y_dim.to(u.m).value
        self.focal_length = focal_length.to(u.m).value
        self.radian_shift = radian_shift
        self.wavelength = wavelength.to(u.m).value
        self.entrance_pixels = e_diam_pixels
        self.only_in_e_diam = only_in_e_diam
        self.pix_per_super = pix_per_super
        
        self.less_than_2pi = less_than_2pi
        
        if self.radian_shift < 2 * np.pi:
            raise Exception('The total phase shift possible is less than 2 pi, so the total phase range available is: ' + str((self.radian_shift - np.pi) / np.pi) + 'pi')
            self.less_than_2pi = True
        
        self.padding_added = None
        self.shift_super_pixel_array = True
        
        self.x_pixscale = x_dim.to(u.m).value/self.pixels_x_orig
        self.y_pixscale = y_dim.to(u.m).value/self.pixels_y_orig
        
        if only_in_e_diam:
            self.dim_x_ratio = self.entrance_pixels // self.pix_per_super + 1
            self.dim_y_ratio = self.entrance_pixels // self.pix_per_super + 1
            
            self.x_dim = self.entrance_pixels * self.x_dim_orig / self.total_x_pixels
            self.y_dim = self.entrance_pixels * self.y_dim_orig / self.total_y_pixels
            
            self.pixels_x = int(self.dim_x_ratio)
            self.pixels_y = int(self.dim_y_ratio)
            
            self.pixels_x_remainder = self.pixels_x_orig - self.pixels_x + 1
            self.pixels_y_remainder = self.pixels_y_orig - self.pixels_y + 1

        else:
            self.x_dim = self.x_dim_orig
            self.y_dim = self.x_dim_orig
            
            self.pixels_x = self.pixels_x_orig
            self.pixels_y = self.pixels_x_orig
            
        self.SLM_ampl = np.ones([self.pixels_y, self.pixels_x])
        self.SLM_phase = np.zeros([self.pixels_y, self.pixels_x])
    
    
    def AddPadding(self, ampl_values = 0, phase_values = 0, verbose = False):
        """
        If the array is only in the e_diam, it adds the padding to make it the correct dimensions

        Parameters
        ----------
        ampl_values : float, [0,1], optional
            What value to pad the amplitude array with. The default is 0.
        phase_values : float, [0,1], optional
            What value to pad the phase array with. The default is 0.
        verbose : Bool, optional
            Wether to output if the padding has already been added or not. The default is False.

        Returns
        -------
        None.

        """
        
        if self.padding_added == True:
            if verbose:    
                print("padding has already been added back to the SLM")
        elif self.only_in_e_diam == False:
            if verbose:
                print("Padding can't be added to an array that is already the SLM dimensions")
        else:
            left   = self.pixels_x_remainder//2
            right  = self.pixels_x_remainder//2 + self.pixels_x_remainder%2
            top    = self.pixels_y_remainder//2
            bottom = self.pixels_y_remainder//2 + self.pixels_y_remainder%2
            
            self.SLM_ampl = np.pad(self.SLM_ampl, ((top, bottom), (left, right)), constant_values = ampl_values)
            self.SLM_phase = np.pad(self.SLM_phase, ((top, bottom), (left, right)), constant_values = phase_values)
            
            self.padding_added = True
            
    def DoublePixelConvert(self, add_padding = True):
        """
        A method that takes the SLM_Ampl and SLM_phase information, and encodes it into
        the SLM_encoded array using the double pixel / super pixel method [WILL ADD CITATION].
        
        Returns
        -------
        SLM_encoded: np.array
            Returns a numpy array with the same dimensions as pixels_x_orig and pixels_y_orig.

        """
        if add_padding:
            self.AddPadding()
        
        self.SLM_ampl[self.SLM_ampl > 1] = 1
        
        self.SLM_ampl[self.SLM_ampl < 0] = 0
        
        
        if self.less_than_2pi:    
            
            self.SLM_phase[self.SLM_phase > (self.radian_shift - .5*np.pi)] = (self.radian_shift - .5*np.pi)
            self.SLM_phase[self.SLM_phase < 0] = 0
        else:
            
            self.SLM_phase[self.SLM_phase > (self.radian_shift)] = (self.radian_shift)
            self.SLM_phase[self.SLM_phase < 0] = 0

        
        SLM_kron_scale = np.ones((self.pix_per_super, self.pix_per_super), dtype = 'bool')
        
        SLM_phase_kron = np.kron(self.SLM_phase, SLM_kron_scale)
        SLM_ampl_kron = np.kron(self.SLM_ampl, SLM_kron_scale)
        
        if add_padding:
            self.SLM_encoded = np.empty([self.total_y_pixels, self.total_x_pixels])
            
            SLM_phase_trimmed = SLM_phase_kron[0:self.total_y_pixels, 0:self.total_x_pixels]
            SLM_ampl_trimmed =  SLM_ampl_kron[0:self.total_y_pixels, 0:self.total_x_pixels]
        else:
            self.SLM_encoded = np.empty((self.entrance_pixels, self.entrance_pixels))
            
            SLM_phase_trimmed = SLM_phase_kron[0:self.entrance_pixels, 0:self.entrance_pixels]
            SLM_ampl_trimmed =  SLM_ampl_kron[0:self.entrance_pixels, 0:self.entrance_pixels]
        
        kron_scale = np.ones((self.pix_per_super//2, self.pix_per_super//2), dtype = 'bool')
        base_indices = np.array([[True,False], [False,True]], dtype = 'bool')
        
        scaled_indices = np.kron(base_indices, kron_scale)
        if add_padding:
            scale_tile_x = self.total_x_pixels // (self.pix_per_super) + 1
            scale_tile_y = self.total_y_pixels // (self.pix_per_super) + 1
        else:
            scale_tile_x = self.dim_x_ratio
            scale_tile_y = self.dim_y_ratio
        
        tiled_array = np.tile(scaled_indices, (scale_tile_y, scale_tile_x))
        
        if add_padding:
            final_indices = tiled_array[0:self.total_y_pixels, 0:self.total_x_pixels]
        else:
            final_indices = tiled_array[0:self.entrance_pixels, 0:self.entrance_pixels]
        
        if self.shift_super_pixel_array == False:
            SLM_ampl_trimmed = np.roll(SLM_ampl_trimmed, (self.x_shift, self.y_shift), axis = (1, 0))
            SLM_phase_trimmed = np.roll(SLM_phase_trimmed, (self.x_shift, self.y_shift), axis = (1, 0))
            final_indices = np.roll(final_indices, (self.x_shift, self.y_shift), axis = (1, 0))
        
        
        self.SLM_encoded[final_indices] = SLM_phase_trimmed[final_indices] + np.arccos(SLM_ampl_trimmed[final_indices])
        self.SLM_encoded[final_indices == False] = SLM_phase_trimmed[final_indices == False] - np.arccos(SLM_ampl_trimmed[final_indices == False])
        
        if self.less_than_2pi == False:
            self.SLM_encoded[self.SLM_encoded < 0] += int(self.radian_shift / (2*np.pi)) * 2 * np.pi
            self.SLM_encoded[self.SLM_encoded > self.radian_shift] -= int(self.radian_shift / (2*np.pi)) * 2 * np.pi
            
        return self.SLM_encoded
    
    def GaussianAmpl(self, a, b_x, b_y, c_x, c_y):
        """
        A method to add a goussian to the amplitude (SLM_ampl) using the equation 
        a * exp((-(y - b_y)**2/c_y) - ((x - b_x)**2/c_x)).

        Parameters
        ----------
        a: float
            The value a in the above equation.
        b_x: float
            The value b_x in the above equation.
        b_y: float
            The value b_y in the above equation.
        c_x: float
            The value c_x in the above equation.
        c_y: float
            The value c_y in the above equation.
        
        Returns
        -------
        None.

        """
        x_grid, y_grid = np.mgrid[0:self.pixels_y, 0:self.pixels_x]
        
        for i in range(len(self.SLM_ampl)):
            for j in range(len(self.SLM_ampl[i])):
                self.SLM_ampl[i,j] += a*np.exp((-(i - b_y)**2/c_y) - ((j - b_x)**2/c_x))
     
    def MaxAmpl(self):
        """
        A method to set the amplitude to the maximum possible value (1).

        Returns
        -------
        None.

        """
        self.SLM_ampl = np.ones([self.pixels_y, self.pixels_x])
        
    def GivenAmpl(self, ampl):
        """
        A method to add a given value to the amplitude.

        Parameters
        ----------
        ampl : float
            The ampltidue to add to the SLM between [0,1].

        Returns
        -------
        None.

        """
        self.SLM_ampl += np.full((self.pixels_y, self.pixels_x), ampl)
        
    def RandomAmpl(self):
        """
        A method to set each double pixel to a random value between [0,1] in SLM_ampl.

        Returns
        -------
        None.

        """
        self.SLM_ampl += np.random.rand(self.pixels_y, self.pixels_x)
        
    def RandomPhase(self):
        """
        A method to set each double pixel to a random value between [0,self.radian_shift] in SLM_ampl

        Returns
        -------
        None.

        """
        self.SLM_phase += self.radian_shift * np.random.rand(self.pixels_y, self.pixels_x)
        
    def InterpolateRandomAmpl(self, num_points, neighbors):
        """
        A method to create a random amplitude map by setting (num_points) points to random values and random positions
        and INterpolating between those points

        Parameters
        ----------
        num_points : int
            The number of random points to use for the interpolation.
        neighbors : int
            THe number of neighbors to use for the interpolation, used to speed up the interpolation (neighbors < numpoints).

        Returns
        -------
        None.

        """
        pixels_x = self.pixels_x
        pixels_y = self.pixels_y
        
        all_x = np.linspace(0, pixels_x-1, num = pixels_x)
        all_y = np.linspace(0, pixels_y-1, num = pixels_y)

        x = random.sample(all_x.tolist(), num_points)
        y = random.sample(all_y.tolist(), num_points)
        
        xyT = np.array([y, x])
        xy = xyT.T
        value = np.random.rand(num_points)

        interpolate = sp.interpolate.RBFInterpolator(xy, value, smoothing = 0, kernel = 'thin_plate_spline', neighbors = neighbors)

        for i in range(pixels_y):
            test = np.full((2,pixels_x), i)
            test[1] = np.linspace(0, pixels_x-1, num = pixels_x)
            inter_temp = interpolate(test.T)
            inter_temp[inter_temp > 1] = 1
            inter_temp[inter_temp < 0] = 0
            self.SLM_ampl[i] += inter_temp
            
    def InterpolateRandomPhase(self, num_points, neighbors):
        """
        A method to create a random phase map by setting (num_points) points to random values and random positions
        and INterpolating between those points

        Parameters
        ----------
        num_points : int
            The number of random points to use for the interpolation.
        neighbors : int
            THe number of neighbors to use for the interpolation, used to speed up the interpolation (neighbors < numpoints).

        Returns
        -------
        None.

        """
        pixels_x = self.pixels_x
        pixels_y = self.pixels_y
        
        all_x = np.linspace(0, pixels_x-1, num = pixels_x)
        all_y = np.linspace(0, pixels_y-1, num = pixels_y)

        x = random.sample(all_x.tolist(), num_points)
        y = random.sample(all_y.tolist(), num_points)
        
        xyT = np.array([y, x])
        xy = xyT.T
        value = np.random.rand(num_points)

        interpolate = sp.interpolate.RBFInterpolator(xy, value, smoothing = 0, kernel = 'thin_plate_spline', neighbors = neighbors)

        for i in range(pixels_y):
            test = np.full((2,pixels_x), i)
            test[1] = np.linspace(0, pixels_x-1, pixels_x)
            inter_temp = interpolate(test.T)
            inter_temp[inter_temp > 1] = 1
            inter_temp[inter_temp < 0] = 0
            self.SLM_phase[i] += inter_temp
       
    def InterpolateRandomBoth(self, num_points, neighbors):
        """
        A method to create a random amplitude and phase map by setting (num_points) points to random values and random positions
        and INterpolating between those points.  The amplitude and phase are mapped to the same values.

        Parameters
        ----------
        num_points : int
            The number of random points to use for the interpolation.
        neighbors : int
            THe number of neighbors to use for the interpolation, used to speed up the interpolation (neighbors < numpoints).

        Returns
        -------
        None.

        """
        pixels_x = self.pixels_x
        pixels_y = self.pixels_y
        
        all_x = np.linspace(0, pixels_x-1, num = pixels_x)
        all_y = np.linspace(0, pixels_y-1, num = pixels_y)

        x = random.sample(all_x.tolist(), num_points)
        y = random.sample(all_y.tolist(), num_points)
        
        xyT = np.array([y, x])
        xy = xyT.T
        value = np.random.rand(num_points)

        interpolate = sp.interpolate.RBFInterpolator(xy, value, smoothing = 0, kernel = 'thin_plate_spline', neighbors = neighbors)

        for i in range(pixels_y):
            test = np.full((2,pixels_x), i)
            test[1] = np.linspace(0, pixels_x-1, num = pixels_x)
            inter_temp = interpolate(test.T)
            inter_temp[inter_temp > 1] = 1
            inter_temp[inter_temp < 0] = 0
            self.SLM_phase[i] = inter_temp
            self.SLM_ampl[i] = inter_temp
    
    def CustomAmpl(self, custom_ampl):
        """
        Adds a custom array to the SLM_ampl array.

        Parameters
        ----------
        custom_ampl : np.array
            A numpy array to add to the SLM_ampl.  The array must have the same dimensions as SLM_ampl

        Returns
        -------
        None.

        """
        self.SLM_ampl += custom_ampl
        
    def CustomPhase(self, custom_phase):
        """
        Adds a custom array to the SLM_phase array.

        Parameters
        ----------
        custom_phase : np.array
            A numpy array to add to the SLM_phase.  The array must have the same dimensions as SLM_phase.

        Returns
        -------
        None.

        """
        self.SLM_phase += custom_phase
        
    def ResetBoth(self):
        """
        Resets the SLM_phase and SLM_ampl arrays back to 0.

        Returns
        -------
        None.

        """
        self.SLM_ampl = np.zeros([self.pixels_y, self.pixels_x])
        self.SLM_phase = np.zeros([self.pixels_y, self.pixels_x])
        
    def ImageShift(self, x_shift, y_shift, shift_super_pixel_array = True):
        """
        Rolls the image a number of pixels in the x and y directions

        Parameters
        ----------
        x_shift : int
            The number of pixels to roll the SLM_ampl and SLM_phase in the x direction (positive is to the left, negative is to the right).
        y_shift : int
            The number of pixels to roll the SLM_ampl and SLM_phase in the x direction (positive is to the left, negative is to the right).
        shift_super_pixel_array: Bool, Optional
            Shifts the super pixel array if True, and shifts the SLM pixel array if False.  This adjusts whether x_shift and y_shift are in super pixel scale or in SLM pixel scale

        Returns
        -------
        None.

        """
        self.shift_super_pixel_array = shift_super_pixel_array
        self.x_shift = x_shift
        self.y_shift = y_shift
        
        if shift_super_pixel_array:
            self.SLM_ampl = np.roll(self.SLM_ampl, (x_shift, y_shift), axis = (1, 0))
            self.SLM_phase = np.roll(self.SLM_phase, (x_shift, y_shift), axis = (1, 0))
        
        
    def ZernikeTerms(self, j_list, j_scaling):
        """
        This function overwrites the SLM phase with a combination of zernike terms.

        Parameters
        ----------
        j_list : list of ints
            The zernike term(s) to index over, uses the Noll index.
        j_scaling : list of floats
            The scaling factor for each of the corresponding Noll terms.

        Returns
        -------
        None.

        """
        
        
        if type(j_list) != 'list' and type(j_scaling) != 'list':
            pass
        elif len(j_list) > len(j_scaling):
            print("The number of indices and scaling factors don't match. Will add additional scaling factors of 1 if indix list is longer.  Will ignore additional scaling factors")
            diff = len(j_list) - len(j_scaling)
            j_scaling.extend([1] * diff)
        elif len(j_list) < len(j_scaling):
            print("The number of indices and scaling factors don't match. Will add additional scaling factors of 1 if indix list is longer.  Will ignore additional scaling factors")
        
        # This section sizes the zernike array to the diameter of the pupil
        if self.pixels_x <= self.pixels_y:
            pixels = self.pixels_x
        else:
            pixels = self.pixels_y
        
        remainder_top = (self.pixels_y - pixels)//2
        remainder_left = (self.pixels_x - pixels)//2
        remainder_bottom = ((self.pixels_y - pixels)//2) + ((self.pixels_y - pixels)%2)
        remainder_right = ((self.pixels_x - pixels)//2) + ((self.pixels_x - pixels)%2)
        
        FinalZernikeArray = np.full((pixels, pixels), 2 * np.pi)

        for i in range(len(j_list)):
            FinalZernikeArray += j_scaling[i] * poppy.zernike.zernike1(j_list[i], npix = pixels, noll_normalize=False)
           
        FinalZernikeArray = np.nan_to_num(FinalZernikeArray, nan = np.nanmin(FinalZernikeArray))
        FinalZernikeArray = np.pad(FinalZernikeArray, ((remainder_top, remainder_bottom),(remainder_left, remainder_right)), constant_values = 1/2 * np.pi)
        
        zernikeMin = FinalZernikeArray.min()
        
        FinalZernikeArray -= zernikeMin
        
        """
        zernikeMax = FinalZernikeArray.max()
        zernikeMin = FinalZernikeArray.min()
        
        while zernikeMax >= 2 * np.pi and zernikeMin < 0 * np.pi:
            zernikeMax = FinalZernikeArray.max()
            zernikeMin = FinalZernikeArray.min()
            
            FinalZernikeArray[FinalZernikeArray >= 2 * np.pi] -= 2*np.pi
            FinalZernikeArray[FinalZernikeArray < 0] += 2*np.pi
        
        """
        #FinalZernikeArray += 0.5 * np.pi
        ScaledZernike = (FinalZernikeArray)
        
        self.SLM_phase = ScaledZernike
            
    def FlatPhase(self, n):
        """
        Adds a flat phase offset to SLM_phase

        Parameters
        ----------
        n : float
            The phase offset to add between [0,1].  1 corresponds to the maximum phase shift, given by (radian_shift - np.pi).

        Returns
        -------
        None.

        """
        self.SLM_phase += np.full((self.pixels_y, self.pixels_x), n)
        
    def FocalPlaneImage(self, FocalArray, PSF_pixscale_x, PSF_pixscale_y, max_or_set_intensity = False, *args):
        """
        A method to encode a certain Focal array into the SLM_ampl and SLM_phase

        Parameters
        ----------
        FocalArray : np.array
            The requested PSF to encode into the SLM.
        PSF_pixscale_x : astropy.units.Quantity, convertable to radians
            The scale of each pixel in the x direction (ONLY in units u.radians or convertable, NOT u.radians/u.pixels).
        PSF_pixscale_y : astropy.units.Quantity, convertable to radians
            The scale of each pixel in the x direction (ONLY in units u.radians or convertable, NOT u.radians/u.pixels).
        max_or_set_intensity : bool, optional
            Whether or not to scale the total intensity to the maximum possible (False), or to the requested intensity (True)
        max_intensity : float, optional
            The total intenisty requested in the focal plane
        Returns
        -------
        None.

        """
        
        if self.radian_shift < 2 * np.pi: # and self.radian_shift != 2 * np.pi:
            raise Exception('the total phase shift of the SLM is not enough to create a focal plane image')
        
        x_FA_dim = (1 * self.wavelength) / PSF_pixscale_x.to(u.rad).value# * len(FocalArray[0])) #* self.focal_length
        y_FA_dim = (1 * self.wavelength) / PSF_pixscale_y.to(u.rad).value# * len(FocalArray)) #* self.focal_length
        
        FA_pixscale_x = x_FA_dim / len(FocalArray[0])
        FA_pixscale_y = y_FA_dim / len(FocalArray)
        
        SLM_PSF_pixscale_x = (self.wavelength) / self.x_dim#.to(u.rad).value
        SLM_PSF_pixscale_y = (self.wavelength) / self.y_dim#.to(u.rad).value
        
        x_PSF_scale = PSF_pixscale_x.to(u.rad).value / SLM_PSF_pixscale_x
        y_PSF_scale = PSF_pixscale_y.to(u.rad).value / SLM_PSF_pixscale_y
        
        x_SLM_scale = FA_pixscale_x / self.x_pixscale
        y_SLM_scale = FA_pixscale_y / self.y_pixscale
        
        
        if self.x_dim - x_FA_dim >=0: 
            x_over = self.x_dim - x_FA_dim
            x_pad = int(x_over/FA_pixscale_x) + 1
        else: 
            x_over = 0
            x_pad = 0
        
        if self.y_dim - y_FA_dim >=0: 
            y_over = self.y_dim - y_FA_dim
            y_pad = int(y_over/FA_pixscale_y) + 1
        else: 
            y_over = 0
            y_pad = 0
        
        if x_over + y_over > 0:
            print('cropping')
            FocalArray = np.pad(FocalArray, ((y_pad, y_pad),(x_pad, x_pad)), constant_values=0)
        
        if x_PSF_scale >= 1 or y_PSF_scale >=1:
            print('rescaling')
            resized_transform_real = rescale(FocalArray.real, (y_PSF_scale, x_PSF_scale))
            resized_transform_imag = rescale(FocalArray.imag, (y_PSF_scale, x_PSF_scale))
            FocalArray = resized_transform_real + resized_transform_imag*1j
            
            #cnt_y = FocalArray.shape[0]//2
            #cnt_x = FocalArray.shape[1]//2
            
            #bottom = cnt_y-int((self.pixels_y//2))
            #top = cnt_y+int((self.pixels_y//2))
            #left = cnt_x-int((self.pixels_x//2))
            #right = cnt_x+int((self.pixels_x//2))
            
        #else:
            #cnt_y = FocalArray.shape[0]//2
            #cnt_x = FocalArray.shape[1]//2
            
            #bottom = cnt_y-int((self.pixels_y//2)//y_scale)
            #top = cnt_y+int((self.pixels_y//2)//y_scale)
            #left = cnt_x-int((self.pixels_x//2)//x_scale)
            #right = cnt_x+int((self.pixels_x//2)//x_scale)
        
        #print(bottom)
        #print(top)
        #print(left)
        #print(right)
        
        #FocalArray = FocalArray[bottom:top, left:right]
        
        """
        #fig1
        plt.figure()
        plt.title('1st plot in focal plane code')
        plt.imshow(np.abs(FocalArray))
        plt.show()
        """
        #center_ft = FocalArray.shape
        
        FocalArray_rolled = np.fft.fftshift(FocalArray) #, shift = ((center_ft[0] -  1)//2, (center_ft[1] -  1)//2), axis = (0,1))
        transform = fft.fft2(FocalArray_rolled)
        total_intensity = transform[0,0]
        transformRolled = np.fft.fftshift(transform)#, shift = (-(center_ft[0] -  1)//2, -(center_ft[1] -  1)//2), axis = (0,1))
        
        if max_or_set_intensity:
            intensity_scale =  args[0] / total_intensity
            if intensity_scale >= 1:
                raise Exception("The total intensity requested is greater than the total intensity present in the focal plane")
        else:
            intensity_scale = 1
        
        """
        plt.figure()
        plt.title('FocalArray rolled in focal plane code')
        plt.imshow(np.abs(FocalArray_rolled))
        plt.show()
        """
        #fig2
        #plt.figure()
        #plt.imshow(np.abs(transformRolled))
        #plt.show()
        
        #plt.figure()
        #plt.imshow(np.angle(transformRolled), cmap = "twilight")
        #plt.show()
        
        resized_transform_real = rescale(transformRolled.real, (y_SLM_scale, x_SLM_scale))
        resized_transform_imag = rescale(transformRolled.imag, (y_SLM_scale, x_SLM_scale))
        resized_transform = resized_transform_real + resized_transform_imag*1j
        """
        plt.figure()
        plt.title('2nd plot in focal plane code')
        plt.imshow(np.abs(transformRolled))
        plt.show()
        """
        cnt_y = resized_transform.shape[0]/2 - 0.5
        cnt_x = resized_transform.shape[1]/2 - 0.5
        
        #print(cnt_y)
        #print(cnt_x)
        
        bottom = int(cnt_y-self.pixels_y/2) + 2
        top = int(cnt_y+self.pixels_y/2) + 2
        left = int(cnt_x-self.pixels_x/2) + 2
        right = int(cnt_x+self.pixels_x/2) + 2
        
        #print(bottom)
        #print(top)
        #print(self.pixels_y)
        
        input_field = resized_transform[bottom:top, left:right]
        """
        #fig3
        plt.figure()
        plt.title('3rd plot in focal plane code')
        plt.imshow(np.abs(input_field))
        plt.show()
        """
        self.transformAmpl = np.abs(input_field)
        self.transformPhase = np.angle(input_field)
        
        transformedAmplMax = self.transformAmpl.max()
        transformedAmplMin = self.transformAmpl.min()
        
        ScaledTransformedAmpl = intensity_scale * (self.transformAmpl - transformedAmplMin) / (transformedAmplMax - transformedAmplMin)
        """
        for i in range(len(self.transformPhase)):
            for j in range(len(self.transformPhase[0])):
                
                if (i+j)%2 == 1 and self.transformPhase[i,j] < 0:
                    self.transformPhase[i,j] += np.pi
                    
                elif (i+j)%2 == 1 and self.transformPhase[i,j] > 0:
                    self.transformPhase[i,j] -= np.pi
        """
        
        self.transformPhase += 1 * np.pi
        
        """
        if self.radian_shift == 2*np.pi:
            self.transformPhase[self.transformPhase > np.pi] == - self.transformPhase[self.transformPhase > np.pi] + 2 * np.pi
        """
        
        self.SLM_ampl = ScaledTransformedAmpl
        self.SLM_phase = self.transformPhase
        #fig4
        #figure = plt.figure()
        #plt.imshow(self.SLM_ampl)
        #plt.show()

        
    #def StatisticalPSDWFE(self, wavelength, index, diameter):
    #    """
    #    UNUSED METHOD, would be used to replicate atmosphereic turbulence
    #
    #    Parameters
    #    ----------
    #    wavelength : TYPE
    #        DESCRIPTION.
    #    index : TYPE
    #        DESCRIPTION.
    #    diameter : TYPE
    #        DESCRIPTION.
    #
    #    Returns
    #    -------
    #    None.
    #
    #    """
    #    pass
    
    def CheckerboardPhaseOnly(self, num_pixels_merged, phaseoffset_max, phaseoffset_min = 0):
        
        self.FlatPhase(0)
        
        base_array = np.array([[True,False], [False,True]], dtype = 'bool')
        kron_scale = np.ones((num_pixels_merged, num_pixels_merged), dtype = 'bool')
        scaled_kron = np.kron(base_array, kron_scale)
        
        y_len = 2 * self.pixels_y//num_pixels_merged + 1
        x_len = 2 * self.pixels_x//num_pixels_merged + 1
        
        tiled_array = np.tile(scaled_kron, (y_len, x_len))
        array_indices = tiled_array[0:self.pixels_y, 0:self.pixels_x]
        
        
        self.SLM_phase[array_indices] = phaseoffset_max
        self.SLM_phase[array_indices == False] = phaseoffset_min
    
    
    def LPModeEncoding(self, N_modes, el, m, n_core, n_cladding, make_odd = False, oversample = 3, oversize = 1):
        """
        A method to encode specific LP modes into the PSF assuming a lens with focal length focal_length.

        Parameters
        ----------
        N_modes : int
            The number of modes in the input MMF at the specified wavelength.
        el : int
            The l number of the LP mode to be encoded, starting at 0.
        m : int
            THe m number of the LP mode to be encoded, starting at 1.
        n_core : float
            The refractive index of the core of the MMF.
        n_cladding : float
            The refractive index of the cladding of the MMF.
        make_odd : bool, optional
            Find the odd or even version of the specific LP mode (False is even, True is odd). The default is False.
        oversample : float, optional
            A scaling factor for the number of pixels to use when creating the LP modes. The default is 1.

        Returns
        -------
        None.

        """
        
        ovsp = oversample
        
        V = np.sqrt(2 * N_modes)
        self.a = (self.wavelength * V) / (2 * np.pi * ofiber.numerical_aperture(n_core, n_cladding)) 
        b = ofiber.LP_mode_value(V, el, m)
        
        
        self.Amplitude = np.zeros((int(oversize*ovsp*self.pixels_y),int(oversize*ovsp*self.pixels_x)))
        self.Phase = np.zeros((int(oversize*ovsp*self.pixels_y),int(oversize*ovsp*self.pixels_x)))

        center_x = (oversize*ovsp*self.pixels_x - 1)/2
        center_y = (oversize*ovsp*self.pixels_y - 1)/2
        
        x_linspace = np.linspace(0, int(oversize*ovsp*self.pixels_x), int(oversize*ovsp*self.pixels_x), endpoint=False)
        y_linspace = np.linspace(0, int(oversize*ovsp*self.pixels_y), int(oversize*ovsp*self.pixels_y), endpoint=False)
        
        xx, yy = np.meshgrid(x_linspace, y_linspace)
        
        x_scale = (xx - center_x) * self.focal_length * np.tan(self.wavelength/(self.x_dim * 10 * ovsp))
        
        y_scale = (yy - center_y) * self.focal_length * np.tan(self.wavelength/(self.y_dim * 10 * ovsp))
        
        r = np.sqrt(x_scale**2 + y_scale**2)
        phi = np.arctan2(y_scale, x_scale)
        
        r_over_a = r/self.a
        """
        plt.figure()
        plt.title('radius from center?')
        plt.imshow(r_over_a)
        plt.show()
        """
        if make_odd: important_bits = ofiber.LP_radial_field(V, b, el, r_over_a) * np.sin(el * phi)
        else: important_bits = ofiber.LP_radial_field(V, b, el, r_over_a) * np.cos(el * phi)
        
        self.Amplitude = np.abs(important_bits)
        
        self.Phase[important_bits >= 0] = np.pi
          
        self.Amplitude /= np.sqrt(np.sum(self.Amplitude**2))
        """
        plt.figure()
        plt.title('original in LP encoding code')
        plt.imshow(self.Amplitude)
        plt.show()
        
        """
        self.fourier_encode = self.Amplitude * np.exp(1j * self.Phase)
        
        pixscale_x = self.wavelength/(self.x_dim * 10 * ovsp) * u.rad
        pixscale_y = self.wavelength/(self.y_dim * 10 * ovsp) * u.rad
        
        self.FocalPlaneImage(self.fourier_encode, pixscale_x, pixscale_y)
        
    def focal_spot(self, central_spot_power, left_spot_power, right_spot_power, spacing, rotation):
        
        Y, X = np.mgrid[-self.pixels_y/2:self.pixels_y/2, -self.pixels_x/2:self.pixels_x/2]
        
        Xr = np.cos(rotation / 180 * np.pi) * X + np.sin(rotation / 180 * np.pi) * Y
        
        period = self.pixels_x / spacing
        
        im_c = left_spot_power * np.exp(-1j * 2 * np.pi / period * Xr) + \
               right_spot_power * np.exp(1j * 2 * np.pi / period * Xr) + central_spot_power
               
        if np.max(np.abs(im_c)) >= 1:
            self.SLM_ampl = np.abs(im_c)/np.max(np.abs(im_c))
        else:
            self.SLM_ampl = np.abs(im_c)
        self.SLM_phase = np.angle(im_c)
    
    def focal_spots_multiple(self, spot_power, spacing, rotation):
        
        self.im_c = np.zeros((self.pixels_y, self.pixels_x), dtype = 'complex')
        
        for i in range(len(spot_power)):
            Y, X = np.mgrid[-self.pixels_y//2:self.pixels_y//2, -self.pixels_x//2:self.pixels_x//2]
            
            Xr = np.cos(rotation[i] / 180 * np.pi) * X + np.sin(rotation[i] / 180 * np.pi) * Y
            
            if spacing[i] == 0:
                temp_c = spot_power[i]
            else:
                period = self.pixels_x / spacing[i]
                temp_c = spot_power[i] * np.exp(-1j * 2 * np.pi / period * Xr)
            
            self.im_c += temp_c
                   
        if np.max(np.abs(self.im_c)) >= 1:
            self.SLM_ampl = np.abs(self.im_c)/np.max(np.abs(self.im_c))
        else:
            self.SLM_ampl = np.abs(self.im_c)
            
        self.SLM_phase = np.angle(self.im_c) + np.pi
            
