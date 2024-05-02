# To simulate data, takes a sample array and translates it small amounts and adds different amounts of noise 

import configparser
from utils import fcns
from astropy.io import fits
import numpy as np
import scipy

stem = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/'
stem_write = stem + 'fake_data/broadband_lamps/'

# true data
test_frame = fcns.read_fits_file(stem + 'sample_data/sample_data_3_spec.fits')
test_data_slice = test_frame[0,:,:]
test_variance_slice = test_frame[1,:,:]

# loop over number of frames to generate
num_frames = 20

for i in range(0,num_frames):

    frame_init = test_frame # will be overwritten
    
    # small x,y offsets
    '''
    xoff = np.random.normal()
    yoff = np.random.normal()
    '''

    # large x,y offsets
    xoff = 10.*np.random.normal()
    yoff = 10.*np.random.normal()

    test_data_slice_shifted = scipy.ndimage.shift(test_data_slice, (-yoff, -xoff), mode='nearest')
    test_variance_slice_shifted = scipy.ndimage.shift(test_data_slice, (-yoff, -xoff), mode='nearest')

    # add some white noise
    test_data_slice_shifted += 5.*np.random.normal(np.shape(test_data_slice_shifted)[0],np.shape(test_data_slice_shifted)[1])
    test_variance_slice_shifted += 1.*np.random.normal(np.shape(test_variance_slice_shifted)[0],np.shape(test_variance_slice_shifted)[1])

    frame_init[0,:,:] = test_data_slice_shifted
    frame_init[1,:,:] = test_variance_slice_shifted

    # save
    file_name = stem_write + f'fake_data_{i:02d}.fits'
    fits.writeto(file_name, frame_init, overwrite=True)
    print('Wrote',file_name)
    