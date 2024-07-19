for col in range(0, x_extent): 
    
    # initialize matrices; we will solve for
    # c_mat.T * x.T = b_mat.T to get x
    c_mat = np.zeros((len(eta_flux), len(eta_flux)), dtype='float')
    b_mat = np.zeros((len(eta_flux)), dtype='float')

    # equivalent variables for variance
    c_mat_prime = np.zeros((len(vark), len(vark)), dtype='float')
    b_mat_prime = np.zeros((len(vark)), dtype='float')

    # loop over pixels in col
    for pix_num in range(0, y_extent):

        # vectorized form of Sharp and Birchall 2010, Eqn. 9 (c_mat is c_kj matrix; b_mat is b_j matrix)
        # (this is equivalent to a for loop over rows of the c_matrix, enclosing a for loop over all spectra (or, equivalently, across all cols of the c_matrix)
        c_mat += dict_profiles_array[:, pix_num, col, np.newaxis] * dict_profiles_array[:, pix_num, col, np.newaxis].T / array_variance[pix_num, col]

        # b_mat is just 1D, so use mat_row as index
        b_mat += D[pix_num, col] * dict_profiles_array[:, pix_num, col] / array_variance[pix_num, col]

        # equivalent expressions for variance, Sharp and Birchall 2010, Eqn. 19 (c_mat_prime is c'_kj matrix; b_mat is b'_j matrix)
        # (note we are treating sqrt(var(Di))=sigmai in Sharp and Birchall's notation)
        c_mat_prime += dict_profiles_array[:, pix_num, col, np.newaxis] * dict_profiles_array[:, pix_num, col, np.newaxis].T
        b_mat_prime += ( array_variance[pix_num, col] - n_rd**2 ) * dict_profiles_array[:, pix_num, col]

    # solve for the following transform:
    # x * c_mat = b_mat  -->  c_mat.T * x.T = b_mat.T
    # we want to solve for x, which is equivalent to spectral flux matrix eta_flux_mat (eta_k in Eqn. 9)
    eta_flux_mat_T, istop, itn, normr, normar, norma, conda, normx = \
            lsmr(c_mat.transpose(), b_mat.transpose())
    
    eta_flux_mat = eta_flux_mat_T.transpose()
    
    for eta_flux_num in range(0, len(eta_flux)):
        eta_flux[str(eta_flux_num)][col] = eta_flux_mat[eta_flux_num]

    ##########
    # solve for same transform again, to get the variance (Eqn. 19 in Sharp and Birchall)
    # x * c_mat_prime = b_mat_prime  -->  c_mat_prime.T * x.T = b_mat_prime.T
    # (we want to solve for x, which is equivalent to variance matrix var_mat (var_k in Eqn. 9)
    var_mat_T, istop, itn, normr, normar, norma, conda, normx = \
            lsmr(c_mat_prime.transpose(), b_mat_prime.transpose())
    
    var_mat = var_mat_T.transpose()
    
    for var_num in range(0, len(eta_flux)): # variance vector is same length as the flux vector
        vark[str(var_num)][col] = var_mat[var_num]

# update class variables
target_instance.spec_flux = eta_flux
target_instance.var_spec_flux = vark