import matplotlib.pyplot as plt
import numpy as np
from utils import fcns, backbone_classes
import time
import configparser

## ## TBD: make clearer distinction between length of spectra, and that of extraction profile

if __name__ == "__main__":

    start_time = time.time()

    stem = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/'

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

    # initialize basic spectrum object which contains spectra info
    spec_obj = backbone_classes.SpecData(num_spec = len(abs_pos_00), 
                                         len_spec = np.shape(test_data_slice)[1], 
                                         sample_frame = test_data_slice)

    # instantiate extraction machinery
    extractor = backbone_classes.Extractor(num_spec = len(abs_pos_00), 
                                           len_spec = np.shape(test_data_slice)[1])
    
    # generate a profile for each spectrum, and update the spec_obj with them
    ## ## TODO: SUBSUME THE test_data_slice INTO THE SPEC_OBJ
    ## ## TODO: SOME OF THE LENGTHY ARGUMENT LISTS OF THESE FUNCTION CALLS CAN PROBABLY BE PRUNED
    extractor.stacked_profiles(target_instance=spec_obj,
                                        abs_pos=abs_pos_00)

    import ipdb; ipdb.set_trace()  

    # do the actual spectral extraction
    extractor.extract_spectra(target_instance=spec_obj,
                                          D=test_data_slice, 
                                          array_variance=test_variance_slice)


    print(type(spec_obj.spec_flux))
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

    for i in range(0,len(spec_obj.spec_flux)):
        plt.plot(spec_obj.spec_flux[str(i)])
        plt.show()

