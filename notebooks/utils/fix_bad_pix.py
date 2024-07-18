# Fix bad pixels

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel, interpolate_replace_nans
#import sys
import glob, os
import multiprocessing

class FixPixSingle:
    '''
    Interpolates over bad pixels
    '''
    
    def __init__(self, write_dir, master_bad_pix_file, config):
        
        self.write_dir = write_dir
        self.abs_badpixmask_name = master_bad_pix_file
        self.config = config
    
        # read in bad pixel mask
        self.badpix, self.header_badpix = fits.getdata(self.abs_badpixmask_name, 0, header=True)
        
        # turn 1->nan (bad), 0->1 (good) for interpolate_replace_nans
        self.ersatz = np.nan*np.ones(np.shape(self.badpix))
        self.ersatz[self.badpix == 0] = 1.
        self.badpix = self.ersatz # rename
        del self.ersatz
    
        # define the convolution kernel (normalized by default)
        self.kernel = np.ones((3,3)) # just a patch around the kernel
    
    def __call__(self, abs_sci_name):
        '''
        Bad pix fixing, for a single frame so as to parallelize job
                                                               
        INPUTS:
        sci_name: science array filename
        '''

        # read in the science frame from raw data directory
        sci, header_sci = fits.getdata(abs_sci_name, 0, header=True)

        # fix bad pixels
        sci_badnan = np.multiply(sci,self.badpix)
        image_fixpixed = interpolate_replace_nans(array=sci_badnan, kernel=self.kernel)

        # write file out
        abs_image_fixpixed_name = str(self.write_dir + os.path.basename(abs_sci_name))
        fits.writeto(filename=abs_image_fixpixed_name,
                     data=image_fixpixed,
                     overwrite=True)
        print("Wrote out bad-pixel-fixed frame " + str(self.write_dir + os.path.basename(abs_sci_name)))