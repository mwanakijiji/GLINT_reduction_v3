import os
import shutil
import time

# copies FITS files into another directory, to mimic the writing of real data

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