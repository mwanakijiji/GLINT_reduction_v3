import ipdb
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from astropy.visualization import ZScaleInterval

# Makes thumbnail PNGs of GLINT engineering data from 2024 Sept.

stem_20240917 = '/mnt/sdata/20240917/apapane/'
stem_20240918 = '/mnt/sdata/20240918/apapane/'

file_list_20240917_1 = glob.glob(stem_20240917 + 'apapane_09*.fits')
file_list_20240917_2 = glob.glob(stem_20240917 + 'apapane_10*.fits')
file_list_20240917_3 = glob.glob(stem_20240917 + 'apapane_11*.fits')

file_list_20240918_1 = glob.glob(stem_20240918 + 'apapane_06*.fits')
file_list_20240918_2 = glob.glob(stem_20240918 + 'apapane_07*.fits')
file_list_20240918_3 = glob.glob(stem_20240918 + 'apapane_09*.fits')
file_list_20240918_4 = glob.glob(stem_20240918 + 'apapane_10*.fits')

file_list_20240917 = file_list_20240917_1 + file_list_20240917_2 + file_list_20240917_3 
file_list_20240918 = file_list_20240918_1 + file_list_20240918_2 + file_list_20240918_3 + file_list_20240918_4

# Create 'thumbnails' directories if they doesn't exist
thumbnails_dir_20240917 = 'thumbnails/20240917'
if not os.path.exists(thumbnails_dir_20240917):
    os.makedirs(thumbnails_dir_20240917)
thumbnails_dir_20240918 = 'thumbnails/20240918'
if not os.path.exists(thumbnails_dir_20240918):
    os.makedirs(thumbnails_dir_20240918)

# Loop through each file, take a median of the cube, and save as a PNG
for file in file_list_20240917:
    print('Opening',file)
    with fits.open(file) as hdul:
        # take median of the file along the long axis
        file_median = np.median(hdul[0].data, axis=0)
        # Save the median image as a PNG
        # Compute the zscale interval
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(file_median)

        # Display the image with the zscale interval
        plt.imshow(file_median, cmap='gray', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(f'Median of 20240917\n{os.path.basename(file)}')
        file_png_name = os.path.join(thumbnails_dir_20240917, os.path.basename(file).replace('.fits', '.png'))
        plt.savefig(file_png_name)
        plt.close()
        print('Wrote',file_png_name)

# Loop through each file, take a median of the cube, and save as a PNG
for file in file_list_20240918:
    print('Opening',file)
    with fits.open(file) as hdul:
        # take median of the file along the long axis
        file_median = np.median(hdul[0].data, axis=0)
        # Save the median image as a PNG
        # Compute the zscale interval
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(file_median)

        # Display the image with the zscale interval
        plt.imshow(file_median, cmap='gray', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(f'Median of 20240918\n{os.path.basename(file)}')
        file_png_name = os.path.join(thumbnails_dir_20240918, os.path.basename(file).replace('.fits', '.png'))
        plt.savefig(file_png_name)
        plt.close()
        print('Wrote',file_png_name)