import pandas as pd
import numpy as np
import ipdb
import scipy
import astropy.io.fits as fits
from scipy.optimize import curve_fit
from scipy.sparse.linalg import lsmr
from astropy.convolution import interpolate_replace_nans
import glob

def apply_wavel_solns(num_spec, source_instance, target_instance):
    '''
    Take wavelength fit coefficients from the source_instance, and use them to map wavelengths to target_instance
    '''

    for i in range(0,num_spec):

        a_coeff = source_instance.fit_coeffs[str(i)][0]
        b_coeff = source_instance.fit_coeffs[str(i)][1]
        f_coeff = source_instance.fit_coeffs[str(i)][2]

        X = (target_instance.spec_x_pix[str(i)], target_instance.spec_y_pix[str(i)])

        target_instance.wavel_mapped[str(i)] = wavel_from_func(X, a_coeff, b_coeff, f_coeff)

    return None


def stacked_profiles(target_instance, abs_pos, len_spec, profiles_file_name=None, sigma=1):
    '''
    Generates a dictionary of profiles based on (x,y) starting positions of spectra
    (This is basically a quick way to set up simple simple horizontal profiles)

    Args:
        target_instance (object): The instance of the Extractor class.
        len_spec: the length of the spectra, in pixels
        abs_pos (dict): A dictionary containing the (x,y) starting positions of spectra.
        profiles_file_name: the file name of the profiles to use (if None, then a simple set of profiles is generated)
        sigma (float, optional): The standard deviation of the Gaussian profile, in pixel units. Defaults to 1.

    Returns:
        None; the value of variable target_instance.dict_profiles is updated
    '''

    array_shape = np.shape(target_instance.sample_frame)
        
    dict_profiles = {}

    if profiles_file_name is None:
        for spec_num in range(0,len(abs_pos)):

            # these profiles are exactly horizontal
            dict_profiles[str(spec_num)] = simple_profile(array_shape = array_shape, 
                                                                    x_left=abs_pos[str(spec_num)][0], 
                                                                    y_left=abs_pos[str(spec_num)][1], 
                                                                    len_profile=len_spec, 
                                                                    sigma_pass=sigma)
        
        # add a little bit of noise for troubleshooting
        #dict_profiles[str(spec_num)] += (1e-3)*np.random.rand(np.shape(dict_profiles[str(spec_num)])[0],np.shape(dict_profiles[str(spec_num)])[1])

        # store the (x,y) values of this spectrum
        ## TBD: improve this later by actually following the spine of the profile
        ## ... and allow for fractional pixels
        target_instance.spec_x_pix[str(spec_num)] = np.arange(array_shape[1])
        target_instance.spec_y_pix[str(spec_num)] = float(abs_pos[str(spec_num)][1]) * np.ones(array_shape[0])

    else:
        # read in profiles
        profiles_data = fits.open(profiles_file_name)
        profiles = profiles_data[0].data

        for spec_num in range(0,len(abs_pos)):

            # these profiles are exactly horizontal
            dict_profiles[str(spec_num)] = profiles[spec_num,:,:]
        
        # store the (x,y) values of this spectrum
        ## TO DO: improve this later by actually following the spine of the profile
        ## ... and allow for fractional pixels
        target_instance.spec_x_pix[str(spec_num)] = np.arange(array_shape[1])
        target_instance.spec_y_pix[str(spec_num)] = None # float(abs_pos[str(spec_num)][1]) * np.ones(array_shape[0])

    target_instance.dict_profiles = dict_profiles

    return


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


def write_to_file(target_instance, file_write):

    # set column definitions
    coldefs_flux = fits.ColDefs([ 
        fits.Column(name=f'spec_{str(i).zfill(2)}_flux', format='D', array=target_instance.spec_flux[str(i)], unit=f'Flux', disp='F8.3')
        for i in target_instance.spec_flux
        ])
    coldefs_var = fits.ColDefs([ 
        fits.Column(name=f'spec_{str(i).zfill(2)}_var', format='D', array=target_instance.vark[str(i)], unit=f'Flux', disp='F8.3')
        for i in target_instance.spec_flux
        ])
    try: # if there is a wavelength mapping; TO DO: put in ersatz here if there isnt one
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
    except:
        print('No wavelength mapping being written out')
        coldefs_all = coldefs_flux + coldefs_var
    

    table_hdu = fits.BinTableHDU.from_columns(coldefs_all)

    # Write the table HDU to the FITS file
    table_hdu.writeto(file_write, overwrite=True)
    print('---------------------------------------')
    print('Wrote extracted spectra to',file_write)

    return None


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
def gauss1d_rot(x_left, len_spec, x_pass, lambda_pass, mu_pass, sigma_pass=1, angle_rot=0):
    '''
    x_left: x coord of leftmost pixel of spectrum (y coord is assumed to be mu_pass)
    len_spec: length of spectrum [pix]
    x_pass: grid of y-coords in coord system of input
    lambda_pass: grid of x-coords in coord system of input
    mu_pass: profile center (in x_pass coords)
    sigma_pass: profile width (in x_pass coords)
    angle_rot: rotation angle in degrees
    Returns:
        profile: Gaussian profile
    '''

    # tile the image left and right (otherwise, profiles may end before reaching the edges of a rotated array)
    x_pass_tiled = np.concatenate((x_pass, x_pass, x_pass), axis=1)
    x_pass_tiled_rot = scipy.ndimage.rotate(x_pass_tiled, angle_rot, reshape=False)
    # crop the left and right sides again
    x_rot = x_pass_tiled_rot[:, np.shape(x_pass)[1]:2*np.shape(x_pass)[1]]

    # rotate lambda_pass (but allow cut-off at the ends of the rotated array)
    lambda_rot = scipy.ndimage.rotate(lambda_pass, angle_rot, reshape=False)

    # Define the Gaussian profile using the rotated coordinates
    profile = (1./(sigma_pass*np.sqrt(2.*np.pi))) * np.exp(-0.5 * np.power((x_rot-mu_pass)/sigma_pass, 2.))
    
    # condition for lambda axis to be inside footprint
    lambda_cond = np.logical_and(lambda_rot >= x_left, lambda_rot < x_left+len_spec)

    # mask regions where there is zero signal
    profile *= lambda_cond
    
    # normalize columns of nonzero signal
    profile = np.divide(profile, np.nanmax(profile))
    
    # restore regions of zero signal as zeros (instead of False)
    profile[~lambda_cond] = 0.
    
    return profile


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
def simple_profile_rot(array_shape, x_left, y_left, len_profile, sigma_pass=1, angle_rot=0):
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
    array_profile = gauss1d(x_left=x_left, 
                                len_spec=len_profile, 
                                x_pass=ygrid, 
                                lambda_pass=xgrid, 
                                mu_pass=y_left, 
                                sigma_pass=sigma_pass)
    '''
    array_profile = gauss1d_rot(x_left=x_left, 
                                len_spec=len_profile, 
                                x_pass=ygrid, 
                                lambda_pass=xgrid, 
                                mu_pass=y_left, 
                                sigma_pass=sigma_pass, 
                                angle_rot=angle_rot)
    '''

    #plt.imshow(array_profile)
    #plt.show()
    
    # normalize it such that the marginalization in x (in (x,lambda) space) is 1
    # (with a perfect Gaussian profile in x this is redundant)
    array_profile[:,x_left:x_left+len_profile] = np.divide(array_profile[:,x_left:x_left+len_profile],np.sum(array_profile[:,x_left:x_left+len_profile], axis=0))
    
    return array_profile


# wrapper to make the enclosing profile of a spectrum
def simple_profile(array_shape, x_left, y_left, len_profile, sigma_pass=1):
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
    array_profile = gauss1d(x_left=x_left, len_spec=len_profile, x_pass=ygrid, lambda_pass=xgrid, mu_pass=y_left, sigma_pass=sigma_pass)

    #plt.imshow(array_profile)
    #plt.show()
    
    # normalize it such that the marginalization in x (in (x,lambda) space) is 1
    # (with a perfect Gaussian profile in x this is redundant)
    array_profile[:,x_left:x_left+len_profile] = np.divide(array_profile[:,x_left:x_left+len_profile],np.sum(array_profile[:,x_left:x_left+len_profile], axis=0))
    
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