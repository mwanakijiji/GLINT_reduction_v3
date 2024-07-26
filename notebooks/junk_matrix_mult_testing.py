import numpy as np
import matplotlib.pyplot as plt
import ipdb

'''
def calculate_c_matrix(phi, sigma):
    # Create the diagonal matrix from the inverse of the variances
    sigma_inv_squared = np.diag(1 / sigma**2)
    
    # Perform the matrix multiplication
    c_matrix = np.dot(np.dot(phi, sigma_inv_squared), phi.T)
    
    return c_matrix

# Example usage
phi = np.array([[1, 2], [3, 4], [5, 6]])  # Example phi matrix
sigma = np.array([1, 2])  # Example sigma vector

c_matrix = calculate_c_matrix(phi, sigma)
print(c_matrix)
'''

# --------------------------


# Assume N and M are defined
N, M = 10, 6

# Generate random matrices for demonstration
phi = np.random.rand(N, M)
D = np.random.rand(M)
sigma_squared = np.random.rand(M)

# Compute S^-2
S_inv_squared = 1 / sigma_squared  # Shape: (M,) # CORRECT

# Compute the element-wise product of phi and S^-2
phi_S = phi * S_inv_squared  # Shape: (N, M) - broadcasting S_inv_squared across rows

# Compute C
#C = np.dot(phi.T, phi_S)  # Shape: (M, M)
C = np.dot(phi_S, phi.T)

ipdb.set_trace()

# Compute b
b = np.dot( phi, np.multiply(D, S_inv_squared) )  # np.matmul works too, since one matrix is 1D # CORRECT

# Solve for eta
eta = np.linalg.solve(C, b)  # Shape: (M,)

print("C matrix:\n", C)
print("b vector:\n", b)
print("eta vector:\n", eta)


print(np.shape(eta))

#plt.imshow(eta)
#plt.show()

ipdb.set_trace()
