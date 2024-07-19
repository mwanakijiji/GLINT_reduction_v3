import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse.linalg import lsmr
from utils import fcns
from astropy.io import fits
import glob
import ipdb
import time
from multiprocessing import Pool
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import warnings
from photutils.centroids import centroid_sources
from photutils.centroids import (centroid_1dg, centroid_2dg,
                                 centroid_com, centroid_quadratic)


def worker(variables_to_pass):

    col, eta_flux, vark, dict_profiles_array, D, array_variance, n_rd = variables_to_pass

    c_mat = np.zeros((len(eta_flux), len(eta_flux)), dtype='float')
    b_mat = np.zeros((len(eta_flux)), dtype='float')
    c_mat_prime = np.zeros((len(vark), len(vark)), dtype='float')
    b_mat_prime = np.zeros((len(vark)), dtype='float')

    # loop over rows
    for pix_num in range(0, D.shape[0]):
        c_mat += dict_profiles_array[:, pix_num, col, np.newaxis] * dict_profiles_array[:, pix_num, col, np.newaxis].T / array_variance[pix_num, col]
        b_mat += D[pix_num, col] * dict_profiles_array[:, pix_num, col] / array_variance[pix_num, col]
        c_mat_prime += dict_profiles_array[:, pix_num, col, np.newaxis] * dict_profiles_array[:, pix_num, col, np.newaxis].T
        b_mat_prime += (array_variance[pix_num, col] - n_rd**2) * dict_profiles_array[:, pix_num, col]

    eta_flux_mat_T, _, _, _, _, _, _, _ = lsmr(c_mat.transpose(), b_mat.transpose())
    eta_flux_mat = eta_flux_mat_T.transpose()

    var_mat_T, _, _, _, _, _, _, _ = lsmr(c_mat_prime.transpose(), b_mat_prime.transpose())
    var_mat = var_mat_T.transpose()

    return col, eta_flux_mat, var_mat


def update_results(results, eta_flux, vark):
    for col, eta_flux_mat, var_mat in results:
        for eta_flux_num in range(len(eta_flux)):
            eta_flux[str(eta_flux_num)][col] = eta_flux_mat[eta_flux_num]
        for var_num in range(len(vark)):
            vark[str(var_num)][col] = var_mat[var_num]

    return


class SpecData:
    # holds info about the spectra being extracted

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


class Extractor():
    # contains machinery for doing the extraction

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


    def write_to_file(self, target_instance, file_write):

        # set column definitions
        coldefs_flux = fits.ColDefs([ 
            fits.Column(name=f'spec_{str(i).zfill(2)}_flux', format='D', array=target_instance.spec_flux[str(i)], unit=f'Flux', disp='F8.3')
            for i in target_instance.spec_flux
            ])
        coldefs_var = fits.ColDefs([ 
            fits.Column(name=f'spec_{str(i).zfill(2)}_var', format='D', array=target_instance.vark[str(i)], unit=f'Flux', disp='F8.3')
            for i in target_instance.spec_flux
            ])
        coldefs_wavel = fits.ColDefs([ 
            fits.Column(name=f'spec_{str(i).zfill(2)}_wavel', format='D', array=target_instance.wavel_mapped[str(i)], unit=f'Wavelength (A)', disp='F8.3')
            for i in target_instance.spec_flux
            ])
        coldefs_x_pix = fits.ColDefs([ 
            fits.Column(name=f'spec_{str(i).zfill(2)}_xpix', format='D', array=target_instance.spec_x_pix[str(i)], unit=f'Pix (x)', disp='F8.3')
            for i in target_instance.spec_flux
            ])
        coldefs_y_pix = fits.ColDefs([ 
            fits.Column(name=f'spec_{str(i).zfill(2)}_ypix', format='D', array=target_instance.spec_y_pix[str(i)], unit=f'Pix (y)', disp='F8.3')
            for i in target_instance.spec_flux
            ])
        
        coldefs_all = coldefs_flux + coldefs_var + coldefs_wavel + coldefs_x_pix + coldefs_y_pix

        table_hdu = fits.BinTableHDU.from_columns(coldefs_all)

        # Write the table HDU to the FITS file
        table_hdu.writeto(file_write, overwrite=True)
        print('---------------------------------------')
        print('Wrote extracted spectra to',file_write)

        return None


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
    

    def apply_wavel_solns(self, source_instance, target_instance):
        '''
        Take wavelength fit coefficients from the source_instance, and use them to map wavelengths to target_instance
        '''

        for i in range(0,self.num_spec):

            a_coeff = source_instance.fit_coeffs[str(i)][0]
            b_coeff = source_instance.fit_coeffs[str(i)][1]
            f_coeff = source_instance.fit_coeffs[str(i)][2]

            X = (target_instance.spec_x_pix[str(i)], target_instance.spec_y_pix[str(i)])

            target_instance.wavel_mapped[str(i)] = fcns.wavel_from_func(X, a_coeff, b_coeff, f_coeff)

        return None
    

    def stacked_profiles(self, target_instance, abs_pos, sigma=1):
        '''
        Generates a dictionary of profiles based on (x,y) starting positions of spectra

        Args:
            target_instance (object): The instance of the Extractor class.
            abs_pos (dict): A dictionary containing the (x,y) starting positions of spectra.
            sigma (float, optional): The standard deviation of the Gaussian profile, in pixel units. Defaults to 1.

        Returns:
            None; the value of variable target_instance.dict_profiles is updated
        '''

        array_shape = np.shape(target_instance.sample_frame)
            
        dict_profiles = {}
        for spec_num in range(0,len(abs_pos)):

            # these profiles are exactly horizontal
            dict_profiles[str(spec_num)] = self.simple_profile(array_shape = array_shape, 
                                                                    x_left=abs_pos[str(spec_num)][0], 
                                                                    y_left=abs_pos[str(spec_num)][1], 
                                                                    len_profile=self.len_spec, 
                                                                    sigma_pass=sigma)
            
            # add a little bit of noise for troubleshooting
            #dict_profiles[str(spec_num)] += (1e-3)*np.random.rand(np.shape(dict_profiles[str(spec_num)])[0],np.shape(dict_profiles[str(spec_num)])[1])

            # store the (x,y) values of this spectrum
            ## TBD: improve this later by actually following the spine of the profile
            ## ... and allow for fractional pixels
            target_instance.spec_x_pix[str(spec_num)] = np.arange(array_shape[1])
            target_instance.spec_y_pix[str(spec_num)] = float(abs_pos[str(spec_num)][1]) * np.ones(array_shape[0])

        target_instance.dict_profiles = dict_profiles

        return
    

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
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        if fyi_plot:
            # make a plot showing how the data array and profiles overlap
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(D/np.std(D), norm=LogNorm())
            plt.subplot(1, 2, 2)
            plt.imshow(D/np.std(D) + np.sum(dict_profiles_array, axis=0), norm=LogNorm(vmin=1e-3, vmax=1))
            plt.show()

        # pack variables other than the column number into an object that can be passed to function with multiprocessing
        variables_to_pass = [eta_flux, vark, dict_profiles_array, D, array_variance, n_rd]

        if process_method == 'parallel':
            num_cpus = multiprocessing.cpu_count()
            pool = Pool(num_cpus)
            time_0 = time.time()
            results = pool.map(worker, [(col, *variables_to_pass) for col in range(x_extent)])
            pool.close()
            pool.join()
            update_results(results, eta_flux, vark)
            time_1 = time.time()
            print('---------')
            print('Full array time taken:')
            print(time_1 - time_0)
        elif process_method == 'series':
            time_0 = time.time()
            results = []
            for col_num in range(x_extent):
                col, eta_flux_mat, var_mat = worker([col_num, *variables_to_pass])
                results.append([col, eta_flux_mat, var_mat])
            update_results(results, eta_flux, vark)
            time_1 = time.time()
            print('---------')
            print('Full array time taken:')
            print(time_1 - time_0)

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