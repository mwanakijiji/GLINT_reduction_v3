import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse.linalg import lsmr

class SpecData:

    def __init__(self, num_spec, len_spec):
        self.num_spec = num_spec # number of spectra to extract
        self.dict_profiles = {} # dict of 2D spectral profiles
        self.wavel_soln = {} # dict of wavelength solns
        #self.profiles = {} # profiles of the spectra

        # initialize dict to hold the extracted spectra ('eta_flux')
        self.spec_flux = {str(i): np.zeros(len_spec) for i in range(self.num_spec)}

        # length of the spectra 
        self.len_spec = len_spec

    def rectangle(self):
        # stub
        return


class Extractor():

    def __init__(self, num_spec, len_spec):
        self.num_spec = num_spec # number of spectra to extract
        self.len_spec = len_spec # length of the spectra 

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

    # gaussian profile (kind of confusing: coordinates are (lambda, x), instead of (x,y) )
    def gauss1d(self, x_left, len_profile, x_pass, lambda_pass, mu_pass, sigma_pass=1):
        '''
        x_left: x coord of leftmost pixel of spectrum (y coord is assumed to be mu_pass)
        len_profile: length of spectrum profile [pix]
        x_pass: grid of y-coords in coord system of input
        lambda_pass: grid of x-coords in coord system of input
        mu_pass: profile center (in x_pass coords)
        sigma_pass: profile width (in x_pass coords)
        '''
        
        # condition for lambda axis to be inside footprint
        lambda_cond = np.logical_and(lambda_pass >= x_left, lambda_pass < x_left+len_profile)
        
        #plt.imshow(lambda_cond)
        #plt.show()
        
        # profile spanning entire array
        self.profile = (1./(sigma_pass*np.sqrt(2.*np.pi))) * np.exp(-0.5 * np.power((x_pass-mu_pass)/sigma_pass,2.) )
        
        # mask regions where there is zero signal
        self.profile *= lambda_cond
        
        # normalize columns of nonzero signal
        self.profile = np.divide(self.profile, np.nanmax(self.profile))
        
        # restore regions of zero signal as zeros (instead of False)
        self.profile[~lambda_cond] = 0.
        
        return self.profile


    # wrapper to make the enclosing profile of a spectrum
    def simple_profile(self, array_shape, x_left, y_left, len_profile, sigma_pass=1):
        """
        Make one simple 1D Gaussian profile in x-direction

        Args: 
            array_shape (tuple): shape of the array
            x_left (int): x-coordinate of the leftmost point of the spectrum
            y_left (int): y-coordinate of the leftmost point of the spectrum
            len_profile (int): length of the spectrum profile in the x-direction (pixels)
            sigma_pass (float): sigma width of the profile (pixels)
        
        Returns:
            numpy.ndarray: 2D array representing the profile on the detector
        """

        x_left = int(x_left)
        y_left = int(y_left)

        array_profile = np.zeros(array_shape)

        xgrid, ygrid = np.meshgrid(np.arange(0,np.shape(array_profile)[1]),np.arange(0,np.shape(array_profile)[0]))
        array_profile = self.gauss1d(x_left=x_left, len_profile=len_profile, x_pass=ygrid, lambda_pass=xgrid, mu_pass=y_left, sigma_pass=sigma_pass)
        array_profile = self.gauss1d(x_left=x_left, len_profile=len_profile, x_pass=ygrid, lambda_pass=xgrid, mu_pass=y_left, sigma_pass=sigma_pass)

        #plt.imshow(array_profile)
        #plt.show()
        
        # normalize it such that the marginalization in x (in (x,lambda) space) is 1
        # (with a perfect Gaussian profile in x this is redundant)
        array_profile[:,x_left:x_left+len_profile] = np.divide(array_profile[:,x_left:x_left+len_profile],np.sum(array_profile[:,x_left:x_left+len_profile], axis=0))
        
        return array_profile
    

    def stacked_profiles(self, array_shape, abs_pos):
        '''
        Generates a dictionary of profiles based on (x,y) starting positions of spectra
        '''
            
        dict_profiles = {}
        for spec_num in range(0,len(abs_pos)):
            dict_profiles[str(spec_num)] = self.simple_profile(array_shape = array_shape, 
                                                                    x_left=abs_pos[str(spec_num)][0], 
                                                                    y_left=abs_pos[str(spec_num)][1], 
                                                                    len_profile=self.len_spec, 
                                                                    sigma_pass=1)
            
            # add a little bit of noise for troubleshooting
            #dict_profiles[str(spec_num)] += (1e-3)*np.random.rand(np.shape(dict_profiles[str(spec_num)])[0],np.shape(dict_profiles[str(spec_num)])[1])

        return dict_profiles
    

    def extract_spectra(self, x_extent, y_extent, eta_flux, dict_profiles, D, array_variance):
        """
        Extracts the spectra

        Args:
            x_extent (int): The number of columns in the detector.
            y_extent (int): The number of pixels in each column.
            eta_flux (dict): A dictionary to store the extracted eta_flux values.
            dict_profiles (dict): A dictionary containing profiles for each row.
            D (ndarray): The 2D data array
            array_variance (ndarray): The 2D variances array

        Returns:
            dict: The updated dictionary containing extracted spectra
        """

        # convert dictionary into a numpy array
        dict_profiles_array = np.array(list(dict_profiles.values()))

        # loop over detector cols (which are treated independently)
        for col in range(0, x_extent): 
            
            # initialize matrices; we will solve for
            # c_mat.T * x.T = b_mat.T to get x
            c_mat = np.zeros((len(eta_flux), len(eta_flux)), dtype='float')
            b_mat = np.zeros((len(eta_flux)), dtype='float')

            # loop over pixels in col
            for pix_num in range(0, y_extent):

                # vectorized form of Sharp and Birchall 2010, Eqn. 9 (this is equivalent to a for loop over rows of the c_matrix, enclosing a for loop over all spectra (or, equivalently, across all cols of the c_matrix)
                c_mat += dict_profiles_array[:, pix_num, col, np.newaxis] * dict_profiles_array[:, pix_num, col, np.newaxis].T / array_variance[pix_num, col]

                # b_mat is just 1D, so use mat_row as index
                b_mat += D[pix_num, col] * dict_profiles_array[:, pix_num, col] / array_variance[pix_num, col]

            # solve for the following transform:
            # x * c_mat = b_mat  -->  c_mat.T * x.T = b_mat.T
            eta_flux_mat_T, istop, itn, normr, normar, norma, conda, normx = \
                    lsmr(c_mat.transpose(), b_mat.transpose())
            
            eta_flux_mat =  eta_flux_mat_T.transpose()
            
            for eta_flux_num in range(0, len(eta_flux)):
                eta_flux[str(eta_flux_num)][col] = eta_flux_mat[eta_flux_num]

        return eta_flux

'''
# EXAMPLE

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