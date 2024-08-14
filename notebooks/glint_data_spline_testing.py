import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression
from astropy.io import fits
import ipdb
import json

# This script is parented from template_spline_testing.py, but reads in real GLINT data

# file name of an empirical readout
file_name = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/data/sample_data/datacube_12_channels.fits'
# just use the first slice, and rotate
hdul = fits.open(file_name)
data = np.rot90(hdul[0].data[0])

#plt.imshow(data, vmin=0, vmax=1000, origin='lower')
#plt.show()

# json file defining the ROIs
file_name_json = '/Users/bandari/Documents/git.repos/GLINT_reduction_v3/notebooks/glint_spline_testing.json'
# Read the JSON file
with open(file_name_json, 'r') as f:
    json_data = json.load(f)

ipdb.set_trace()
# Extract the coordinates from the JSON data
rois = json_data['rois']

x_min, y_min = rois['1'][0]
x_max, y_max = rois['1'][1]

ipdb.set_trace()

# Threshold to identify high-value pixels
threshold = 800  # Adjust this threshold based on your data

# Extract coordinates of high-value pixels
high_value_coords = np.argwhere(data > threshold)
x_coords = high_value_coords[:, 1]
y_coords = high_value_coords[:, 0]

# Create a mask for the ROI
roi_mask = (x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max)

# Apply the mask to filter the coordinates
x_coords_roi = x_coords[roi_mask]
y_coords_roi = y_coords[roi_mask]

# Sort coordinates by x to ensure a proper spline fit
sorted_indices = np.argsort(x_coords_roi)
x_coords_sorted = x_coords_roi[sorted_indices]
y_coords_sorted = y_coords_roi[sorted_indices]

m_, b_ = np.polyfit(x_coords_sorted, y_coords_sorted, deg=1)

'''
# linear fit (can provide best results)
x_coords_sorted = x_coords_sorted.reshape(-1, 1)
y_coords_sorted = y_coords_sorted.reshape(-1, 1)
model = LinearRegression()
test = model.fit(x_coords_sorted, y_coords_sorted)
y_pred = model.predict(x_coords_sorted)
'''

'''
# Fit a spline to the high-value pixels
spline = UnivariateSpline(x_coords_sorted, y_coords_sorted, k=1, s=1)
'''

# Generate x values for plotting the spline
x_spline = np.linspace(x_coords_sorted.min(), x_coords_sorted.max(), 1000).reshape(-1, 1)

# y_spline = spline(x_spline) # UniVariate
y_spline = model.predict(x_spline) # Linear regression


## Now make a Gaussian that follows that spline 

# Function to calculate the distance from a point to the spline
'''
def distance_to_spline(x, y, spline):
    y_spline = spline(x)
    return np.abs(y - y_spline)
'''

ipdb.set_trace()

#x_coords, y_coords = np.meshgrid(np.arange(np.shape(data)[1]), np.arange(np.shape(data)[0])

distances = np.abs(m * x_coords + b - y_coords) / np.sqrt(m**2 + 1)


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