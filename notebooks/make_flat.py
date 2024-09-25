import ipdb
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

stem = '/import/morgana1/snert/GLINTData/data202409/20240919/apapane/'

file_list_fiber_in = glob.glob(stem + 'apapane_03:3[0-2]*fits')
file_list_fiber_out = glob.glob(stem + 'apapane_03:3[6-8]*fits')

# Initialize an empty list to store the 2D data arrays
data_cube_fiber_in = []
data_cube_fiber_out = []

# Loop through each file and append the data to the list
for file in file_list_fiber_in:
    print('Opening',file)
    with fits.open(file) as hdul:
        # take median of the file along the long axis
        file_median = np.median(hdul[0].data, axis=0)
        data_cube_fiber_in.append(file_median)

for file in file_list_fiber_out:
    with fits.open(file) as hdul:
        print('Opening',file)
        file_median = np.median(hdul[0].data, axis=0)
        data_cube_fiber_out.append(file_median)

# Convert the list to a 3D numpy array
data_cube_fiber_in = np.array(data_cube_fiber_in)
data_cube_fiber_out = np.array(data_cube_fiber_out)

flat_median_fiber_in = np.mean(data_cube_fiber_in, axis=0)
flat_median_fiber_out = np.mean(data_cube_fiber_out, axis=0)

# Write the median images to FITS files
file_name_1 = 'flat_median_fiber_in.fits'
fits.writeto(file_name_1, flat_median_fiber_in, overwrite=True)
print('Wrote',file_name_1)
file_name_2 = 'flat_median_fiber_out.fits'
fits.writeto(file_name_2, flat_median_fiber_out, overwrite=True)
print('Wrote',file_name_2)