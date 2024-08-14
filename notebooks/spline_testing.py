import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import ipdb

# This is a working script that fits a spline to a series of dots on a 2D array, and makes a Gaussian that 
# lines up with that spline

# Example 2D array
data = 0.1 * np.random.rand(100, 100)  # Replace with your 2D array
# Flag the coordinates which fulfill the condition y = x * 0.2 + 36
x_coords, y_coords = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
#ipdb.set_trace()
flagged_coords = np.where(np.isclose(y_coords, x_coords * 0.2 + 36))
data[flagged_coords] = 1.1


# Threshold to identify high-value pixels
threshold = 0.9  # Adjust this threshold based on your data

# Extract coordinates of high-value pixels
high_value_coords = np.argwhere(data > threshold)
x_coords = high_value_coords[:, 1]
y_coords = high_value_coords[:, 0]

# Sort coordinates by x to ensure a proper spline fit
sorted_indices = np.argsort(x_coords)
x_coords_sorted = x_coords[sorted_indices]
y_coords_sorted = y_coords[sorted_indices]

# Fit a spline to the high-value pixels
spline = UnivariateSpline(x_coords_sorted, y_coords_sorted, s=1)

# Generate x values for plotting the spline
x_spline = np.linspace(x_coords_sorted.min(), x_coords_sorted.max(), 1000)
y_spline = spline(x_spline)


## Now make a Gaussian that follows that spline 

# Function to calculate the distance from a point to the spline
def distance_to_spline(x, y, spline):
    y_spline = spline(x)
    return np.abs(y - y_spline)

# Calculate the distance for each point on the grid
distances = np.zeros_like(data, dtype=float)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        distances[i, j] = distance_to_spline(j, i, spline)

# Parameters for the Gaussian profile
sigma = 5  # Standard deviation of the Gaussian

# Create the Gaussian profile
gaussian_profile = np.exp(-0.5 * (distances / sigma) ** 2)



# Plot the original data and the fitted spline
plt.imshow(data, cmap='gray', origin='lower')
plt.scatter(x_coords, y_coords, color='red', label='High-value pixels')
plt.plot(x_spline, y_spline, color='blue', label='Fitted spline')
plt.legend()
plt.show()

# Plot the Gaussian profile
plt.imshow(gaussian_profile, cmap='hot', origin='lower')
plt.colorbar(label='Gaussian Intensity')
plt.title('2D Gaussian Profile with Peak Along Spline')
plt.show()