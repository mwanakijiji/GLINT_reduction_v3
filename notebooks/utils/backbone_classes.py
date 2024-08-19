import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse.linalg import lsmr
from utils import fcns
from astropy.io import fits
import glob
import ipdb
import time
import multiprocessing
from multiprocessing import Pool
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import warnings
from scipy.linalg import lstsq
from photutils.centroids import centroid_sources
from photutils.centroids import (centroid_1dg, centroid_2dg,
                                 centroid_com, centroid_quadratic)

def rearrange_into_big_2d(matrix_input, P):

    N, M = matrix_input.shape
    # Initialize the larger matrix with zeros
    larger_matrix = np.zeros((P * N, P * M))
    
    # Place matrix_input at the appropriate diagonal positions
    for i in range(P):
        start_row = i * N
        start_col = i * M
        larger_matrix[start_row:start_row + N, start_col:start_col + M] = matrix_input
    
    return larger_matrix

def worker(variables_to_pass):
    '''
    Does detector-array-column specific reductions with matrix math
    '''

    dict_profiles_array, D, array_variance, n_rd = variables_to_pass

    # tile everything, for the same data and extraction

    ### begin tiling test

    # P>1 doesn't work yet
    '''
    D_big = rearrange_into_big_2d(matrix_input=D, P=2)
    array_variance_big = rearrange_into_big_2d(array_variance, P=2)
    dict_profiles_array_big = rearrange_into_big_2d(dict_profiles_array[:,:,0], P=2)

    t0 = time.time()
    phi = dict_profiles_array_big # [:, col]
    D = D_big[:, col]
    array_variance_big = array_variance_big[:, col]
    '''
    #ipdb.set_trace()
    ### end tiling test



    # this chunk works
    '''
    phi = dict_profiles_array[:, :, col]
    D = D[:, col]
    array_variance_big = array_variance[:, col]
    '''

    phi = dict_profiles_array 
 
    array_variance_big = array_variance

    # Compute S^-2
    S_inv_squared = 1 / array_variance_big  # Shape: (M,) # CORRECT

    # Compute the element-wise product of phi and S^-2
    phi_S = phi * S_inv_squared  # Shape: (N, M) - broadcasting S_inv_squared across rows

    #c_matrix_big = np.dot( phi_S, phi.T )
    c_matrix_big = np.einsum('ijk,jlk->ilk', phi_S, np.transpose(phi, (1, 0, 2)))
    #c_matrix_big = np.dot( phi_S, np.transpose(phi, (1, 0, 2)) )

    # Compute b
    #b_matrix_big = np.dot( phi, np.multiply(D, S_inv_squared) )  # np.matmul works too, since one matrix is 1D # CORRECT
    b_matrix_big = np.einsum('ijk,jk->ik', phi, np.multiply(D, S_inv_squared))

    #c_mat_prime = np.dot( phi, phi.T )
    c_mat_prime = np.einsum('ijk,jlk->ilk', phi, np.transpose(phi, (1, 0, 2)))
    #b_mat_prime = np.dot( phi, (array_variance_big - n_rd**2) )
    b_mat_prime = np.einsum('ijk,jk->ik', phi, array_variance_big - n_rd**2)

    # replace non-finite values with a median value to let solution work
    finite_values_c = c_matrix_big[np.isfinite(c_matrix_big)]
    median_value_c = np.median(finite_values_c)
    non_finite_mask_c = ~np.isfinite(c_matrix_big)
    c_matrix_big[non_finite_mask_c] = median_value_c # Replace non-finite values with the median value

    finite_values_b = b_matrix_big[np.isfinite(b_matrix_big)]
    median_value_b = np.median(finite_values_b)
    non_finite_mask_b = ~np.isfinite(b_matrix_big) # mask for the non-finite values
    b_matrix_big[non_finite_mask_b] = median_value_b # Replace non-finite values with the median value

    # Initialize an array to store the results
    # TO DO: GENERALIZE THIS TO FIT THE LENGTH OF THE INPUT SPECTRA
    eta_flux_mat_T_reshaped = np.zeros((3, 320)) # (# spectra, #x-pixels)
    var_mat_T_reshaped = np.zeros((3, 320))

    # loop over cols
    for i in range(np.shape(eta_flux_mat_T_reshaped)[1]):
        try:
            # Solve the least squares problem for each slice
            result_eta, _, _, _ = lstsq(c_matrix_big[:, :, i], b_matrix_big[:, i])
            result_var, _, _, _ = lstsq(c_mat_prime[:, :, i], b_mat_prime[:, i])
            #result_eta, _, _, _ = np.linalg.lstsq(c_matrix_big[:, :, i], b_matrix_big[:, i], rcond=None)
            #result_var, _, _, _ = np.linalg.lstsq(c_mat_prime[:, :, i], b_mat_prime[:, i], rcond=None)
            # Store the result
            eta_flux_mat_T_reshaped[:, i] = result_eta
            var_mat_T_reshaped[:, i] = result_var
        except:
            eta_flux_mat_T_reshaped[:, i] = np.nan * np.ones(12)
            var_mat_T_reshaped[:, i] = np.nan * np.ones(12)

    
    return eta_flux_mat_T_reshaped, var_mat_T_reshaped


def update_results(results, eta_flux, vark):
    for col, eta_flux_mat, var_mat in results:
        for eta_flux_num in range(len(eta_flux)):
            eta_flux[str(eta_flux_num)][col] = eta_flux_mat[eta_flux_num]
        for var_num in range(len(vark)):
            vark[str(var_num)][col] = var_mat[var_num]

    return


class SpecData:
    # holds info specific to the spectra being extracted from one detector readout

    def __init__(self, num_spec, len_spec, sample_frame):
        self.num_spec = num_spec # number of spectra to extract
        self.dict_profiles = {} # dict of 2D spectral profiles

        self.sample_frame = sample_frame # an example frame, which will be used for getting dims, etc.
        #self.profiles = {} # profiles of the spectra

        # initialize dict to hold the extracted spectra ('eta_flux')
        self.spec_flux = {str(i): np.zeros(len_spec) for i in range(self.num_spec)}
        # dict to hold the extracted variances ('vark')
        self.vark = {str(i): np.zeros(len_spec) for i in range(self.num_spec)}
        # dict to hold the extracted spectra pixel x-vals
        self.spec_x_pix = {str(i): np.zeros(len_spec) for i in range(self.num_spec)}
        # dict to hold the extracted spectra pixel y-vals
        self.spec_y_pix = {str(i): np.zeros(len_spec) for i in range(self.num_spec)}
        # dict to hold the wavelength soln coeffs
        #self.fit_coeffs = {}
        # dict to hold the mapped wavelength abcissae
        self.wavel_mapped = {str(i): np.zeros(len_spec) for i in range(self.num_spec)}

        # length of the spectra 
        self.len_spec = len_spec


class GenWavelSoln:
    # contains machinery for generating wavelength solution

    def __init__(self, num_spec, dir_read, wavel_array):
        self.num_spec = num_spec
        self.dir_read = dir_read # directory containing basis set images
        self.wavel_array = wavel_array # array of wavelengths (any units) in order of sorted filenames
        self.fit_coeffs = {} # dict to hold the wavelength soln coeffs
        self.wavel_soln_data = {str(i): {'x_pix_locs': np.zeros(len(wavel_array)),
                                         'y_pix_locs': np.zeros(len(wavel_array))} for i in range(self.num_spec)} # dict to hold the (x,y) of the wavelength soln basis set

    def make_basis_cube(self):

        file_list = sorted(glob.glob(self.dir_read + '*.fits'))
        cube = np.stack([fits.getdata(file) for file in file_list], axis=0)

        return cube
    

    def find_xy_narrowbands(self, xy_guesses, basis_cube):
        """
        Find the (x,y) values of points in each narrowband frame.

        Parameters:
        - xy_guesses: dictionary containing the initial (x,y) guesses for each narrowband frame
        - basis_cube: 3D numpy array containing the narrowband frames

        Returns:
        None
        """

        # loop over spectra
        for key in xy_guesses:

            for slice_num in range(0,len(basis_cube[:,0,0])):

                data = basis_cube[slice_num,:,:]
                x, y = centroid_sources(data, 
                                        xy_guesses[key]['x_guesses'][slice_num], 
                                        xy_guesses[key]['y_guesses'][slice_num], 
                                        box_size=5,
                                centroid_func=centroid_com)
            
                self.wavel_soln_data[key]['x_pix_locs'][slice_num] = x
                self.wavel_soln_data[key]['y_pix_locs'][slice_num] = y

        return None
    

    # read in a lamp basis image
    def add_basis_image(self, file_name):

        self.lamp_basis_frame = fcns.read_fits_file(file_name)

        return None


    # take input (x,y,lambda) values and do a polynomial fit
    def gen_coeffs(self, target_instance):
        '''
        Generate the coefficients for the (x,y,lambda) data

        target_instance: the instance to which the wavelength solution will be mapped
        '''

        # wavel_soln_dict: dictionary of three arrays of floats:
        # - x_pix_locs: x-locations of empirical spot data on detector (can be fractional)
        # - y_pix_locs: y-locations " " 
        # - lambda_pass: wavelengths corresponding to spots
        wavel_soln_dict = self.wavel_soln_data

        # for each spectrum, take the (xs, ys, lambdas) and generate coeffs (a,b,c)
        for i in range(0,self.num_spec):

            x_pix_locs = wavel_soln_dict[str(i)]['x_pix_locs']
            y_pix_locs = wavel_soln_dict[str(i)]['y_pix_locs']
            lambda_pass = self.wavel_array

            # fit coefficients based on (x,y) coords of given spectrum and the set of basis wavelengths
            fit_coeffs = fcns.find_coeffs(x_pix_locs, y_pix_locs, lambda_pass)

            target_instance.fit_coeffs[str(i)] = fit_coeffs

        return None

#class ExtractorObservingBlock():
    # contains machinery for doing the extraction which is NOT necessary to 
    # initialize with each detector readout


class Extractor():
    # contains machinery for doing the extraction

    def __init__(self, num_spec, len_spec):
        self.num_spec = num_spec # number of spectra to extract
        self.len_spec = len_spec # length of the spectra 
        num_cpus = multiprocessing.cpu_count()
        self.pool = Pool(num_cpus)
    

    def extract_spectra(self, target_instance, D, array_variance, n_rd=0, process_method = 'series', fyi_plot=False):
        """
        Extracts the spectra

        Args:
            target_instance (object): The instance to which variables will be modified.
            x_extent (int): The number of columns in the detector.
            y_extent (int): The number of pixels in each column.
            eta_flux (dict): A dictionary to store the extracted eta_flux values.
            dict_profiles (dict): A dictionary containing profiles for each row.
            D (ndarray): The 2D data array
            array_variance (ndarray): The 2D variances array
            process_method: should detector columns be reduced in parallel? choices 'series' or 'parallel'
            n_rd: amount of read noise

        Returns:
            dict: The updated dictionary containing extracted spectra
        """

        # set values of some things based on the target instance
        x_extent = np.shape(target_instance.sample_frame)[1]
        y_extent = np.shape(target_instance.sample_frame)[0]
        eta_flux = target_instance.spec_flux
        vark = target_instance.vark
        dict_profiles = target_instance.dict_profiles

        # convert dictionary into a numpy array
        dict_profiles_array = np.array(list(dict_profiles.values()))

        # silence warnings about non-convergence
        #warnings.filterwarnings("ignore", category=RuntimeWarning)

        if fyi_plot:
            # make a plot showing how the data array and profiles overlap
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(D/np.std(D), norm=LogNorm())
            plt.subplot(1, 2, 2)
            plt.imshow(D/np.std(D) + np.sum(dict_profiles_array, axis=0), norm=LogNorm(vmin=1e-3, vmax=1))
            plt.show()

        # pack variables other than the column number into an object that can be passed to function with multiprocessing
        variables_to_pass = [dict_profiles_array, D, array_variance, n_rd]

        # treat the columns in series or in parallel?
        if process_method == 'series':
            time_0 = time.time()

            # list comprehension over all the columns
            eta_results, var_results = worker([*variables_to_pass])

            time_1 = time.time()
            print('---------')
            print('Full array time taken:')
            print(time_1 - time_0)

            
            # update the spectral object
            target_instance.spec_flux = {str(i): eta_results[i, :] for i in range(eta_results.shape[0])}
            target_instance.vark = {str(i): var_results[i, :] for i in range(var_results.shape[0])}

        else:
            print('No parallel method available right now.')
