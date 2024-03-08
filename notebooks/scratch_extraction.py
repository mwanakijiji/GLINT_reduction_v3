import matplotlib.pyplot as plt
import numpy as np
from utils import basic_fcns
import time




##################################
## ## BEGIN TBD: INCORPORATE THIS
'''
class Extractor:
    def __init__(self, array):
        self.array = array

    def rectangle(self, start_x, start_y, end_x, end_y):
        self.rectangle = self.array[start_y:end_y, start_x:end_x]

    def sum_along_axis(self, axis):
        return np.sum(self.rectangle, axis=axis)
    
    def plot_1d_array(self, array1):
        plt.plot(array1)
        plt.show()
    
    def plot_2d_array_with_rectangle(self, array2, start_x, start_y, end_x, end_y):
        plt.imshow(array2, origin='lower')
        plt.plot([start_x, start_x, end_x, end_x, start_x], [start_y, end_y, end_y, start_y, start_y], 'r')
        plt.show()

# Create an instance of the Extractor class using the test_1_slice variable
extractor = Extractor(test_data_slice)

# Define the coordinates for the rectangle
start_x = 30
start_y = 95
end_x = 300
end_y = 110

# Call the rectangle method of the Extractor instance
extractor.rectangle(start_x, start_y, end_x, end_y)

# Call the sum_along_axis method of the Extractor instance
sum_along_x = extractor.sum_along_axis(axis=0)
sum_along_y = extractor.sum_along_axis(axis=1)

# Plot the array with the rectangle
#extractor.plot_2d_array_with_rectangle(test_data_slice, start_x, start_y, end_x, end_y)

#extractor.plot_1d_array(sum_along_x)
#extractor.plot_1d_array(sum_along_y)

# Print the results
print("Sum along x-axis:", sum_along_x)
print("Sum along y-axis:", sum_along_y)
'''
## ## END TBD: INCORPORATE THIS
##################################



if __name__ == "__main__":

    start_time = time.time()

    stem = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/'

    # true data
    test_frame = basic_fcns.read_fits_file(stem + 'sample_data_3_spec.fits')
    test_data_slice = test_frame[0,:,:]
    test_variance_slice = test_frame[1,:,:]

    #import ipdb; ipdb.set_trace()

    # fake data
    '''
    test_frame = basic_fcns.read_fits_file('test_array.fits')
    test_data_slice = test_frame
    test_variance_slice = np.sqrt(test_data_slice)
    # insert some noise
    test_data_slice += (1e-3)*np.random.rand(np.shape(test_data_slice)[0],np.shape(test_data_slice)[1])
    test_variance_slice += (1e-3)*np.random.rand(np.shape(test_variance_slice)[0],np.shape(test_variance_slice)[1])
    '''
    
    #import ipdb; ipdb.set_trace()


    # initialize dictionaries which will store the spectrum profiles
    ## THIS SETS THE NUMBER OF SPECTRA TO BE EXTRACTED
    abs_pos, eta_flux, wavel_solns = basic_fcns.infostack(x_extent=np.shape(test_data_slice)[1], 
                                                          y_extent=np.shape(test_data_slice)[0])

    dict_profiles = {}

    # generate a profile for each spectrum
    for spec_num in range(0,len(eta_flux)):
        dict_profiles[str(spec_num)] = basic_fcns.simple_profile(array_shape = np.shape(test_data_slice), 
                                                                 x_left=abs_pos[str(spec_num)][0], 
                                                                 y_left=abs_pos[str(spec_num)][1], 
                                                                 len_spec=200, 
                                                                 sigma_pass=1)
        # add a little bit of noise for troubleshooting
        #dict_profiles[str(spec_num)] += (1e-3)*np.random.rand(np.shape(dict_profiles[str(spec_num)])[0],np.shape(dict_profiles[str(spec_num)])[1])
        
    
        

    #import ipdb; ipdb.set_trace()
    # extract the spectra and put them into dictionary
    eta_flux = basic_fcns.extract_spectra(x_extent=np.shape(test_data_slice)[1], 
                                          y_extent=np.shape(test_data_slice)[0], 
                                          eta_flux=eta_flux, 
                                          dict_profiles=dict_profiles, 
                                          D=test_data_slice, 
                                          array_variance=test_variance_slice)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

    for i in range(0,len(eta_flux)):
        plt.plot(eta_flux[str(i)])
        plt.show()

