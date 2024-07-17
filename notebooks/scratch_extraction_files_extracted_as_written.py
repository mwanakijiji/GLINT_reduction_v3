import matplotlib.pyplot as plt
import numpy as np
from utils import fcns, backbone_classes
import time
import os
import configparser
from configparser import ExtendedInterpolation
from astropy.io import fits
import glob
import image_registration
from image_registration import chi2_shift
from image_registration.fft_tools import shift
import json
import time

## ## TBD: make clearer distinction between length of spectra, and that of extraction profile

# Requirements:
# Within a parent directory, put subdirectories with the string 'series' in the name
# Within each of those subdirectories, put FITS frames which contain in the file name... 
# ... 'data': these are data frames
# ... 'broadband': these are the lamp frames (there should just be one of these in each subdirectory)

if __name__ == "__main__":

    start_time = time.time()

    # Read the config file
    config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
    config.read('config.ini')

    # make directories if they don't exist yet
    for key, value in config['sys_dirs'].items():
        #directory = os.path.join(stem, value)
        os.makedirs(value, exist_ok=True)

    # directory containing files to 'extract'
    dir_spectra_parent = config['sys_dirs']['DIR_DATA'] # fake data made from real
    # Glob the directories inside the specified directory
    dir_spectra_read = glob.glob(dir_spectra_parent + '*series*/')
    # directory to which we will write spectral solutions
    dir_spectra_write = config['sys_dirs']['DIR_WRITE']

    # wavelength configuration stuff
    with open(config['file_names']['FILE_NAME_WAVELXYGUESS'], 'r') as file:
        data = json.load(file)
    
    xy_guesses_basis_set = data['xy_guesses_basis_set'] # array of spots corresponding to narrowband spots
    wavel_array = data['wavel_array'] # array of sampled wavelengths
    abs_pos_00 = data['abs_pos_00'] # spectrum starting positions in the frame we consider to be the basis (absolute coordinates, arbit. number of spectra)

    # a sample frame (to get dims etc.)
    test_frame = fcns.read_fits_file(config['file_names']['FILE_NAME_SAMPLE'])
    test_data_slice = test_frame[0,:,:]
    test_variance_slice = test_frame[1,:,:]

    # fake data for quick checks: uncomment this section to get spectra which are the same as the profiles
    '''
    test_frame = fcns.read_fits_file('test_array.fits')
    test_data_slice = test_frame
    test_variance_slice = np.sqrt(test_data_slice)
    # insert some noise
    test_data_slice += (1e-3)*np.random.rand(np.shape(test_data_slice)[0],np.shape(test_data_slice)[1])
    test_variance_slice += (1e-3)*np.random.rand(np.shape(test_variance_slice)[0],np.shape(test_variance_slice)[1])
    '''

    # generate the basis wavelength solution
    wavel_gen_obj = backbone_classes.GenWavelSoln(num_spec = len(abs_pos_00), 
                                                  dir_read = config['sys_dirs']['DIR_WAVEL_DATA'], 
                                                  wavel_array = np.array(wavel_array))

    basis_cube = wavel_gen_obj.make_basis_cube()

    # find (x,y) of narrowband (i.e., point-like) spectra in each frame of basis cube
    wavel_gen_obj.find_xy_narrowbands(xy_guesses = xy_guesses_basis_set,
                                      basis_cube = basis_cube)
    
    # generate solution coefficients
    wavel_gen_obj.gen_coeffs(target_instance=wavel_gen_obj)

    # read in a lamp basis image (to find offsets later)
    wavel_gen_obj.add_basis_image(file_name = config['file_names']['FILE_NAME_BASISLAMP'])

    # retrieve lamp image
    lamp_file_name = glob.glob(config['file_names']['FILE_NAME_THISLAMP'])
    lamp_data = fits.open(lamp_file_name[0]) # list of files should just have one element
    lamp_array_this = lamp_data[0].data[0]

    # find offset from lamp basis image
    xoff, yoff, exoff, eyoff = chi2_shift(wavel_gen_obj.lamp_basis_frame, lamp_array_this)
    
    # loop over all groups of calibration lamp / data files

    # Get the initial list of files in the directory
    initial_files = os.listdir(dir_spectra_parent)

    # Start monitoring the directory for new files
    while True:
        # Get the current list of files in the directory
        current_files = os.listdir(dir_spectra_parent)

        # Find the new files that have appeared
        new_files = [file for file in current_files if file not in initial_files]

        # Process the new files
        for file in new_files:
            # Construct the full path to the file
            file_path = os.path.join(dir_spectra_parent, file)

            # read in image
            hdul = fits.open(file_path)

            readout_data = hdul[0].data[0,:,:]
            readout_variance = hdul[0].data[1,:,:]

            # translate the image to align it with the basis lamp (i.e., with the wavelength solns)
            readout_data = shift.shiftnd(readout_data, (-yoff, -xoff))
            readout_variance = shift.shiftnd(readout_variance, (-yoff, -xoff))

            # initialize basic spectrum object which contains spectra info
            spec_obj = backbone_classes.SpecData(num_spec = len(abs_pos_00), 
                                                len_spec = np.shape(test_data_slice)[1], 
                                                sample_frame = test_data_slice)

            # instantiate extraction machinery
            extractor = backbone_classes.Extractor(num_spec = len(abs_pos_00),
                                                len_spec = np.shape(test_data_slice)[1])
            
            # generate a profile for each spectrum, and update the spec_obj with them
            extractor.stacked_profiles(target_instance=spec_obj,
                                                abs_pos=abs_pos_00)

            # do the actual spectral extraction, and update the spec_obj with them
            extractor.extract_spectra(target_instance=spec_obj,
                                                D=readout_data, 
                                                array_variance=readout_variance, 
                                                n_rd=0, 
                                                fyi_plot=False)

            # apply the wavelength solution
            extractor.apply_wavel_solns(source_instance=wavel_gen_obj, target_instance=spec_obj)

            # write to file
            file_name_write = dir_spectra_write + 'extracted_' + os.path.basename(file_path)
            extractor.write_to_file(target_instance=spec_obj, file_write = file_name_write)

            end_time = time.time()
            execution_time = end_time - start_time
            print("Execution time:", execution_time, "seconds")

            # loop over all spectra on that detector frame
            if config['options']['WRITE_PLOTS']:

                for i in range(0,len(spec_obj.spec_flux)):

                    # plot the spectra
                    file_name_plot = config['sys_dirs']['DIR_WRITE_FYI'] + os.path.basename(file_path).split('.')[0] + '.png'
                    plt.clf()
                    plt.plot(spec_obj.wavel_mapped[str(i)], spec_obj.spec_flux[str(i)], label='flux')
                    plt.plot(spec_obj.wavel_mapped[str(i)], np.sqrt(spec_obj.vark[str(i)]), label='$\sqrt{\sigma^{2}}$')
                    plt.legend()
                    plt.savefig( file_name_plot )
                    print('Wrote',file_name_plot)

        # Update the initial list of files
        initial_files = current_files

        # Wait for some time before checking again
        time.sleep(1)