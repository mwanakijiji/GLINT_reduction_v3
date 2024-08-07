import pandas as pd
import numpy as np
import astropy.io.fits as fits
from scipy.optimize import curve_fit
from scipy.sparse.linalg import lsmr
from astropy.convolution import interpolate_replace_nans
import glob

def fix_bad(array_pass, badpix_pass):
    '''
    Fixes bad pixels

    Assumes bad pixel mask has 0: good, 1: bad
    '''
    kernel_square = np.ones((3,3)) # correction kernel
    array_pass[badpix_pass == 1] = np.nan
    frame_fixed = interpolate_replace_nans(array=array_pass, kernel=kernel_square) # replace nans

    return frame_fixed

# initial guesses for wavelength solution polynomial coeffs a,b,c:
def find_coeffs(x,y,z):
    '''
    x: x [pix]
    y: y [pix]
    z: wavel
    '''

    p0 = 1., 1., 1., # initial guess
    fit_coeffs, cov = curve_fit(wavel_from_func, (x,y), z, p0)

    return fit_coeffs


def open_fits_table(file_name, verbose=False):

    hdul = fits.open(file_name)

    if verbose:
        print(hdul.info())
        data = hdul[1].data
        print(data.columns)

    return hdul



# return wavelength based on a pre-existing polynomial wavelength solution
def wavel_from_func(X, a_coeff, b_coeff, f_coeff):
    '''
    functional fit to lambda as fcn of (x,y)

    X: (x,y) array
    a,b,c,d,f: coefficients
    '''

    x_pass, y_pass = X

    #return a_coeff + b_coeff*x_pass + c_coeff*y_pass + d_coeff*np.multiply(x_pass,y_pass) + f_coeff*np.power(x_pass,2.) + g_coeff*np.power(y_pass,2.)

    # no y-dependency for the moment
    return a_coeff + b_coeff*x_pass + f_coeff*np.power(x_pass,2.)


def read_fits_file(file_path):
    try:
        hdul = fits.open(file_path)
        # Access the data or header information
        data_array = hdul[0].data
        hdul.close()
    except Exception as e:
        print("Error reading FITS file:", str(e))

    return data_array


# gaussian profile (kind of confusing: coordinates are (lambda, x), instead of (x,y) )
def gauss1d(x_left, len_spec, x_pass, lambda_pass, mu_pass, sigma_pass=1):
    '''
    x_left: x coord of leftmost pixel of spectrum (y coord is assumed to be mu_pass)
    len_spec: length of spectrum [pix]
    x_pass: grid of y-coords in coord system of input
    lambda_pass: grid of x-coords in coord system of input
    mu_pass: profile center (in x_pass coords)
    sigma_pass: profile width (in x_pass coords)
    '''
    
    # condition for lambda axis to be inside footprint
    lambda_cond = np.logical_and(lambda_pass >= x_left, lambda_pass < x_left+len_spec)
    
    #plt.imshow(lambda_cond)
    #plt.show()
    
    # profile spanning entire array
    profile = (1./(sigma_pass*np.sqrt(2.*np.pi))) * np.exp(-0.5 * np.power((x_pass-mu_pass)/sigma_pass,2.) )
    
    # mask regions where there is zero signal
    profile *= lambda_cond
    
    # normalize columns of nonzero signal
    profile = np.divide(profile, np.nanmax(profile))
    
    # restore regions of zero signal as zeros (instead of False)
    profile[~lambda_cond] = 0.
    
    return profile


# wrapper to make the enclosing profile of a spectrum
def simple_profile(array_shape, x_left, y_left, len_spec, sigma_pass=1):
    """
    Make one simple 1D Gaussian profile in x-direction

    Args: 
        array_shape (tuple): shape of the array
        x_left (int): x-coordinate of the leftmost point of the spectrum
        y_left (int): y-coordinate of the leftmost point of the spectrum
        len_spec (int): length of the spectrum in the x-direction (pixels)
        sigma_pass (float): sigma width of the profile (pixels)
    
    Returns:
        numpy.ndarray: 2D array representing the profile on the detector
    """

    x_left = int(x_left)
    y_left = int(y_left)

    array_profile = np.zeros(array_shape)

    xgrid, ygrid = np.meshgrid(np.arange(0,np.shape(array_profile)[1]),np.arange(0,np.shape(array_profile)[0]))
    array_profile = gauss1d(x_left=x_left, len_spec=len_spec, x_pass=ygrid, lambda_pass=xgrid, mu_pass=y_left, sigma_pass=sigma_pass)

    #plt.imshow(array_profile)
    #plt.show()
    
    # normalize it such that the marginalization in x (in (x,lambda) space) is 1
    # (with a perfect Gaussian profile in x this is redundant)
    array_profile[:,x_left:x_left+len_spec] = np.divide(array_profile[:,x_left:x_left+len_spec],np.sum(array_profile[:,x_left:x_left+len_spec], axis=0))
    
    return array_profile


def gen_spec_profile(rel_pos, x_shift, y_shift, canvas_array, D, df_wavel_empirical_zeroed, len_spec, sigma):
    """
    Generates spectrum profiles and calculates wavelength solutions.

    Args:
        rel_pos (dict): Dictionary of relative positions for each spectrum.
        x_shift (int): X-shift value.
        y_shift (int): Y-shift value.
        canvas_array (ndarray): Array to accumulate profile footprints.
        D (ndarray): 2D data array.
        df_wavel_empirical_zeroed (DataFrame): DataFrame of zeroed empirical wavelengths.
        len_spec (int): Length of spectrum in x-direction.
        sigma (float): Sigma width of profile.

    Returns:
        dict: Dictionary containing all profiles.
    """

    # loop over each spectrum's starting position and 
    # 1. generate a full spectrum profile
    # 2. calculate a wavelength solution
    for key, coord_xy in rel_pos.items():

        spec_x_left = np.add(coord_xy[0],-x_shift)
        spec_y_left = np.add(coord_xy[1],-y_shift)

        # place profile on detector, while removing translation of frame relative to a reference frame
        profile_this_array = simple_profile(array_shape=np.shape(D), 
                                    x_left=spec_x_left, 
                                    y_left=spec_y_left, 
                                    len_spec=len_spec, 
                                    sigma_pass=sigma)
        
        # accumulate these onto an array that will let us look at the total footprint
        canvas_array += profile_this_array

        # save single profiles in an array
        dict_profiles[key] = profile_this_array

    return dict_profiles


def file_list_maker(dir_read):

    files = glob.glob(dir_read + '*.fits')

    return files



def infostack(x_extent, y_extent):
    """
    This function extracts spectra from a given region of interest and returns the absolute positions,
    extracted spectra, and wavelength solutions.

    Args:
        x_extent (int): The extent in x of the spectrum
        y_extent (int): The extent in y of the spectrum (not used yet)

    Returns:
        tuple: A tuple containing the following:
            - abs_pos (dict): A dictionary containing the absolute positions (x,y) of the spectra origins
            - eta_flux (dict): A dictionary containing the extracted spectra (initialized to zeros)
            - wavel_soln_ports (dict): A dictionary containing the wavelength solutions for each spectrum (initialized as an empty dict)
    """

    # spectrum starting positions (absolute coordinates)
    abs_pos = {'0':(0,177),
            '1':(0,159),
            '2':(0,102)}
    
    # dict to hold the extracted spectra
    eta_flux = {'0':np.zeros(x_extent),
        '1':np.zeros(x_extent),
        '2':np.zeros(x_extent)}

    # will hold wavelength solutions, one for each of the spectra
    wavel_solns = {}

    return abs_pos, eta_flux, wavel_solns