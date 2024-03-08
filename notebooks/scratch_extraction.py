import matplotlib.pyplot as plt
import numpy as np
from utils import fcns, backbone_classes
import time

import ipdb; ipdb.set_trace()

if __name__ == "__main__":

    start_time = time.time()

    stem = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/'

    # true data
    test_frame = fcns.read_fits_file(stem + 'sample_data_3_spec.fits')
    test_data_slice = test_frame[0,:,:]
    test_variance_slice = test_frame[1,:,:]

    #import ipdb; ipdb.set_trace()

    # fake data: spectra which are the same as the profiles
    '''
    test_frame = fcns.read_fits_file('test_array.fits')
    test_data_slice = test_frame
    test_variance_slice = np.sqrt(test_data_slice)
    # insert some noise
    test_data_slice += (1e-3)*np.random.rand(np.shape(test_data_slice)[0],np.shape(test_data_slice)[1])
    test_variance_slice += (1e-3)*np.random.rand(np.shape(test_variance_slice)[0],np.shape(test_variance_slice)[1])
    '''
    
    #import ipdb; ipdb.set_trace()


    # initialize dictionaries which will store the spectrum profiles
    ## THIS SETS THE NUMBER OF SPECTRA TO BE EXTRACTED
    abs_pos, eta_flux, wavel_solns = fcns.infostack(x_extent=np.shape(test_data_slice)[1], 
                                                          y_extent=np.shape(test_data_slice)[0])

    dict_profiles = {}

    # generate a profile for each spectrum
    for spec_num in range(0,len(eta_flux)):
        dict_profiles[str(spec_num)] = fcns.simple_profile(array_shape = np.shape(test_data_slice), 
                                                                 x_left=abs_pos[str(spec_num)][0], 
                                                                 y_left=abs_pos[str(spec_num)][1], 
                                                                 len_spec=200, 
                                                                 sigma_pass=1)
        # add a little bit of noise for troubleshooting
        #dict_profiles[str(spec_num)] += (1e-3)*np.random.rand(np.shape(dict_profiles[str(spec_num)])[0],np.shape(dict_profiles[str(spec_num)])[1])
        
    
        

    #import ipdb; ipdb.set_trace()
    # extract the spectra and put them into dictionary
    eta_flux = fcns.extract_spectra(x_extent=np.shape(test_data_slice)[1], 
                                          y_extent=np.shape(test_data_slice)[0], 
                                          eta_flux=eta_flux, 
                                          dict_profiles=dict_profiles, 
                                          D=test_data_slice, 
                                          array_variance=test_variance_slice)

    print(type(eta_flux))
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

    for i in range(0,len(eta_flux)):
        plt.plot(eta_flux[str(i)])
        plt.show()

