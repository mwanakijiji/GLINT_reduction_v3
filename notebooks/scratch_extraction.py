import matplotlib.pyplot as plt
import numpy as np
from utils import fcns, backbone_classes
import time
import configparser
from astropy.io import fits

## ## TBD: make clearer distinction between length of spectra, and that of extraction profile

if __name__ == "__main__":

    start_time = time.time()

    #dir_read = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/'
    dir_read = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/fake_data/'

    # Read the config file
    config = configparser.ConfigParser()
    config.read('config.ini')
    # Get the values from the config file
    #variable1 = config.get('section_name', 'variable1')

    # Use the variables in your code
    # For example:
    # stem = variable1
    # abs_pos_00 = variable2

    # spectrum starting positions (absolute coordinates, arbit. number of spectra)
    abs_pos_00 = {'0':(0,177),
        '1':(0,159),
        '2':(0,102), 
        '3':(0,72)}

    # true data
    stem = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/'
    test_frame = fcns.read_fits_file(stem + 'sample_data_3_spec.fits')
    test_data_slice = test_frame[0,:,:]
    test_variance_slice = test_frame[1,:,:]

    # fake data: spectra which are the same as the profiles
    '''
    test_frame = fcns.read_fits_file('test_array.fits')
    test_data_slice = test_frame
    test_variance_slice = np.sqrt(test_data_slice)
    # insert some noise
    test_data_slice += (1e-3)*np.random.rand(np.shape(test_data_slice)[0],np.shape(test_data_slice)[1])
    test_variance_slice += (1e-3)*np.random.rand(np.shape(test_variance_slice)[0],np.shape(test_variance_slice)[1])
    '''

    # retrieve list of files to operate on
    file_list = fcns.file_list_maker(dir_read)

    # loop over all data files
    for file_num in range(0,len(file_list)):

        # read in image
        hdul = fits.open(file_list[file_num])

        readout_data = hdul[0].data[0,:,:]
        readout_variance = hdul[0].data[1,:,:]

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

        # do the actual spectral extraction
        extractor.extract_spectra(target_instance=spec_obj,
                                            D=readout_data, 
                                            array_variance=readout_variance, 
                                            n_rd=0)


        print(type(spec_obj.spec_flux))
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")

        for i in range(0,len(spec_obj.spec_flux)):
            plt.plot(spec_obj.spec_flux[str(i)], label='flux')
            plt.plot(np.sqrt(spec_obj.vark[str(i)]), label='$\sqrt{\sigma^{2}}$')
            plt.legend()
            plt.show()

