from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

def main():

    # Read the FITS file
    hdul = fits.open(filename)

    # extract table
    table = hdul[1].data

    # plot
    '''
    plt.plot(table['spec_00_wavel'], table['spec_00_flux'])
    plt.plot(table['spec_01_wavel'], table['spec_01_flux'])
    plt.plot(table['spec_02_wavel'], table['spec_02_flux'])
    plt.plot(table['spec_03_wavel'], table['spec_03_flux'])
    plt.plot(table['spec_04_wavel'], table['spec_04_flux'])
    plt.plot(table['spec_05_wavel'], table['spec_05_flux'])
    plt.plot(table['spec_06_wavel'], table['spec_06_flux'])
    plt.show()
    '''

    plt.clf()
    plt.plot(table['spec_00_flux'])
    plt.plot(table['spec_01_flux'])
    plt.plot(table['spec_02_flux'])
    plt.plot(table['spec_03_flux'])
    plt.plot(table['spec_04_flux'])
    plt.plot(table['spec_05_flux'])
    plt.plot(table['spec_06_flux'])
    plt.show()

if __name__ == "__main__":

    filename = '/Users/bandari/Documents/git.repos/glint-control/sharp_birchall_extraction/dataset_12_channel/outputs/extracted_slice_4.fits'
    
    main(filename)

