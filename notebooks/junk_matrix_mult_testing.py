import numpy as np

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