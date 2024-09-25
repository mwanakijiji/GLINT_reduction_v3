import ipdb
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# makes simulated narrowband data for finding a wavelength solution

shape = (640, 512)

#source_file = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/sample_data/datacube_12_channels.fits'
#destination_dir = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/fake_data'

# Read the FITS file
#hdul = fits.open(source_file)
#data = hdul[0].data

# non-redundant array of x-values of spot coords
x_vals = np.array([55, 220, 355])
y_vals = np.array([86, 118, 148, 180, 212, 243, 274, 304, 336, 399, 428, 460])

sigma = 2

basis_set_cube = np.zeros((len(x_vals), shape[0], shape[1]))

# Loop over each slice in the cube
for x_val_num in range(0,len(x_vals)):

    array_slice = np.zeros(shape)

    # Make a 2D Gaussian centered on (50,100)
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    for y_val_num in range(0,len(y_vals)):
        gaussian = np.exp(-((x - x_vals[x_val_num])**2 + (y - y_vals[y_val_num])**2) / (2 * sigma**2))

        # Multiply the Gaussian with the data
        array_slice = array_slice + gaussian

    basis_set_cube[x_val_num,:,:] = array_slice
    
# Create a new HDU with the slice data
hdu = fits.PrimaryHDU(basis_set_cube)

# Save the new FITS file
file_name = 'junk.fits'
hdu.writeto(file_name, overwrite=True)

print('Wrote',file_name)