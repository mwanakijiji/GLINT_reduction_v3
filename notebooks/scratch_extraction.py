import matplotlib.pyplot as plt
import numpy as np
from utils import fcns, backbone_classes
import time
import configparser
from astropy.io import fits
import glob
from image_registration import chi2_shift
from image_registration.fft_tools import shift
import image_registration

## ## TBD: make clearer distinction between length of spectra, and that of extraction profile

if __name__ == "__main__":

    start_time = time.time()

    stem = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/'

    # directory containing files to 'extract'
    dir_spectra_parent = stem + 'fake_data/' # fake data made from real
    # Glob the directories inside the specified directory
    dir_spectra_read = glob.glob(dir_spectra_parent + '*series*/')

    # Read the config file
    config = configparser.ConfigParser()
    config.read('config.ini')
    # Get the values from the config file
    #variable1 = config.get('section_name', 'variable1')

    # Use the variables in your code
    # For example:
    # stem = variable1
    # abs_pos_00 = variable2

    # spectrum starting positions in the frame we consider to be the basis (absolute coordinates, arbit. number of spectra)
    abs_pos_00 = {'0':(0,177),
        '1':(0,159),
        '2':(0,102), 
        '3':(0,72)}

    # a sample frame (to get dims etc.)
    stem = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/sample_data/'
    test_frame = fcns.read_fits_file(stem + 'sample_data_3_spec.fits')
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
    # TBD: make this get generated for each calibration flash lamp, not for every file; only apply for every file
    wavel_gen_obj = backbone_classes.GenWavelSoln(num_spec = len(abs_pos_00), 
                                                  dir_read = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/wavel_3PL_basis_data/', 
                                                  wavel_array = np.array([1020, 1060, 1100, 1140, 1180, 1220, 1260, 1300, 1360, 1380, 1420, 1460, 1500, 1540, 1580, 1620, 1660, 1700, 1740]))

    basis_cube = wavel_gen_obj.make_basis_cube()

    # guesses of narrowband points in wavelength solution basis set
    xy_guesses_basis_set = {'0': {'x_guesses': np.array([7.5, 47.6, 80.9, 111, 137, 158, 178, 196, 211, 226, 239, 250, 261, 271, 280, 287, 295, 303, 310]),'y_guesses': 178.*np.ones(19)},
                          '1': {'x_guesses': np.array([7.5, 47.6, 80.9, 111, 137, 158, 178, 196, 211, 226, 239, 250, 261, 271, 280, 287, 295, 303, 310]),'y_guesses': 159.*np.ones(19)},
                          '2': {'x_guesses': np.array([7.5, 47.6, 80.9, 111, 137, 158, 178, 196, 211, 226, 239, 250, 261, 271, 280, 287, 295, 303, 310]),'y_guesses': 102.*np.ones(19)},
                          '3': {'x_guesses': np.array([7.5, 47.6, 80.9, 111, 137, 158, 178, 196, 211, 226, 239, 250, 261, 271, 280, 287, 295, 303, 310]),'y_guesses': 72.*np.ones(19)}
                          }

    # find (x,y) of narrowband (i.e., point-like) spectra in each frame of basis cube
    wavel_gen_obj.find_xy_narrowbands(xy_guesses = xy_guesses_basis_set,
                                      basis_cube = basis_cube)
    
    # generate solution coefficients
    wavel_gen_obj.gen_coeffs(target_instance=wavel_gen_obj)

    # read in a lamp basis image (to find offsets later)
    wavel_gen_obj.add_basis_image(file_name = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/sample_data/sample_data_3_spec.fits')
    

    # loop over all groups of calibration lamp / data files
    for dir_lamp in dir_spectra_read:

        # retrieve lamp image
        lamp_file_name = glob.glob(dir_lamp + '*broadband*.fits')
        lamp_data = fits.open(lamp_file_name[0]) # list of files should just have one element
        lamp_array_this = lamp_data[0].data[0]

        # find offset from lamp basis image
        xoff, yoff, exoff, eyoff = chi2_shift(wavel_gen_obj.lamp_basis_frame, lamp_array_this)
    
        # retrieve list of data files to operate on (does not include calibration lamps)
        file_list = glob.glob(dir_lamp + '*data*.fits')

        # loop over all data files
        for file_num in range(0,len(file_list)):

            # read in image
            hdul = fits.open(file_list[file_num])

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
                                                fyi_plot=True)

            # apply the wavelength solution
            extractor.apply_wavel_solns(source_instance=wavel_gen_obj, target_instance=spec_obj)

            # write to file
            extractor.write_to_file(target_instance=spec_obj, file_write='junk.fits')

            end_time = time.time()
            execution_time = end_time - start_time
            print("Execution time:", execution_time, "seconds")


            for i in range(0,len(spec_obj.spec_flux)):

                # plot the spectra
                plt.plot(spec_obj.wavel_mapped[str(i)], spec_obj.spec_flux[str(i)], label='flux')
                plt.plot(spec_obj.wavel_mapped[str(i)], np.sqrt(spec_obj.vark[str(i)]), label='$\sqrt{\sigma^{2}}$')
                plt.legend()
                plt.show()


