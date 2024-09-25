import ipdb
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

# Makes thumbnail PNGs of GLINT engineering data from 2024 Sept.

stem = '/import/morgana1/snert/GLINTData/data202409/20240919/apapane/'

file_list = glob.glob(stem + '*fits')

# Create 'thumbnails' directory if it doesn't exist
thumbnails_dir = 'thumbnails'
if not os.path.exists(thumbnails_dir):
    os.makedirs(thumbnails_dir)

# Loop through each file, take a median of the cube, and save as a PNG
for file in file_list:
    print('Opening',file)
    with fits.open(file) as hdul:
        # take median of the file along the long axis
        file_median = np.median(hdul[0].data, axis=0)
        # Save the median image as a PNG
        plt.imshow(file_median, cmap='gray')
        plt.colorbar()
        plt.title(f'Median of {os.path.basename(file)}')
        file_png_name = os.path.join(thumbnails_dir, os.path.basename(file).replace('.fits', '.png'))
        plt.savefig(file_png_name)
        plt.close()
        print('Wrote',file_png_name)