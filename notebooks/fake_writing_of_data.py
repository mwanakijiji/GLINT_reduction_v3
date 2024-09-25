import os
import shutil
import time
import os
import ipdb
import shutil
import glob

from astropy.io import fits

# copies FITS files into another directory, to mimic the writing of real data

'''
# METHOD 1: Copy single FITS frames from one directory to another

source_dir = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/fake_data/escrow'
destination_dir = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/fake_data'

# Get a list of all FITS files in the source directory
fits_files = [file for file in os.listdir(source_dir) if file.endswith('.fits')]

# Copy each FITS file to the destination directory
for file in fits_files:
    source_file = os.path.join(source_dir, file)
    destination_file = os.path.join(destination_dir, file)
    shutil.copy(source_file, destination_file)
    print('Copied file ',file)
    # Insert a sleep of 5 seconds
    time.sleep(5)

print('FITS files copied successfully!')

# Delete each copied file from the destination directory
for file in fits_files:
    destination_file = os.path.join(destination_dir, file)
    os.remove(destination_file)
    print('Deleted file ', file)

print('Copied files deleted successfully!')
'''

# METHOD 2: Take one cube of frames, and write individual slices out to another directory

source_file = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/sample_data/datacube_12_channels.fits'
destination_dir = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/fake_data'

# Read the FITS file
hdul = fits.open(source_file)
data = hdul[0].data

# Loop over each slice in the cube
for i, slice_data in enumerate(data):

    # Create the output file name
    output_file = os.path.join(destination_dir, f'slice_{i}.fits')
    
    # Create a new HDU with the slice data
    hdu = fits.PrimaryHDU(slice_data)
    
    # Save the new FITS file
    hdu.writeto(output_file, overwrite=True)
    
    print(f'Slice {i} written to {output_file}')
    time.sleep(1)

print('All slices written successfully!')
'''

# METHOD 3: Take a bunch of cubes and write out the first slice of each as a separate file

source_dir = '/import/morgana2/snert/VAMPIRESData/20240509/apapane/'
destination_dir = '/import/morgana2/snert/VAMPIRESData/20240509/apapane/fake_data/'

file_list = glob.glob(source_dir + "*.fits")

for file_name in file_list:

    # Read the FITS file
    hdul = fits.open(file_name)
    data = hdul[0].data

    # write out the first slice
    slice_data = data[0,:,:]

    # Create a new HDU with the slice data
    hdu = fits.PrimaryHDU(slice_data)
    
    # Save the new FITS file
    file_name_write = destination_dir + os.path.basename(file_name)
    hdu.writeto(file_name_write, overwrite=True)

    print('Wrote out ',file_name_write)
    time.sleep(1)
'''