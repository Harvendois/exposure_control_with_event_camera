import cv2
import numpy as np
from scipy.spatial.distance import euclidean
import pandas as pd
import matplotlib.pyplot as plt

def adjust_gamma(image, gamma=1.0):
    invGamma = gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def dense_sift(image, step_size=8, grid_size=(4, 4), exclude_coords=None):
    sift = cv2.SIFT_create()
    keypoints = []

    # Generate grid points, but exclude those in exclude_coords
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            if (x, y) not in exclude_coords:
                keypoints.append(cv2.KeyPoint(x, y, step_size))

    # Compute SIFT descriptors at each grid point
    _, descriptors = sift.compute(image, keypoints)

    # Reshape descriptors into a grid structure for easier comparison
    descriptors_grid = descriptors.reshape(len(range(0, image.shape[0], step_size)), 
                                           len(range(0, image.shape[1], step_size)), 
                                           -1)
    return descriptors_grid

def compare_images(image1, image2, step_size=8, exclude_coords=None):
    assert image1.shape == image2.shape, "Images must be the same size."

    descriptors1 = dense_sift(image1, step_size=step_size, exclude_coords=exclude_coords)
    descriptors2 = dense_sift(image2, step_size=step_size, exclude_coords=exclude_coords)

    distances = []
    for i in range(descriptors1.shape[0]):
        for j in range(descriptors1.shape[1]):
            if (i * step_size, j * step_size) not in exclude_coords:
                dist = euclidean(descriptors1[i, j], descriptors2[i, j])
                distances.append(dist)

    comparison_metric = np.mean(distances)
    return comparison_metric

def sum_of_contours(image):
    # Find edges using Canny edge detection
    edges = cv2.Canny(image, 100, 200)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sum of contour lengths
    contour_sum = sum(cv2.arcLength(cnt, True) for cnt in contours)
    return contour_sum

def count_polarity_changes(group):
    polarity_changes = (group['p'].diff().abs() == 1).sum()
    return polarity_changes

# # Load the data from the .txt file
# file_path = './txt/rectified_events_left_0.txt'
# columns = ['x', 'y', 't', 'p']
# data = pd.read_csv(file_path, sep='\s+', header=None, names=columns, skiprows=1, dtype={'x': float, 'y': float, 't': float, 'p': int})

# grouped = data.groupby(['x', 'y'])

# polarity_changes = grouped.apply(count_polarity_changes)

# # Find coordinates with few polarity changes (less than 5)
# threshold = 5
# stable_coordinates = polarity_changes[polarity_changes < threshold]
# exclude_coords = set(zip(stable_coordinates.index.get_level_values(0), stable_coordinates.index.get_level_values(1)))

# print(f"Coordinates to exclude: {exclude_coords}")

exclude_coords = []

# Load images
# voxel_grid_event = cv2.imread('../dsec_dataset_interlaken_00_c/voxel_grid_representations/voxel_grid_left_200.png', cv2.IMREAD_GRAYSCALE)
# voxel_grid_event = cv2.imread('../dsec_dataset_zurich_city_12_a/voxel_grid_representations/voxel_grid_left_50.png', cv2.IMREAD_GRAYSCALE)
voxel_grid_event = cv2.imread('../dsec_dataset_zurich_city_09_b/voxel_grid_representations/voxel_grid_left_50.png', cv2.IMREAD_GRAYSCALE)
# gray_image = cv2.imread('../dsec_dataset_interlaken_00_c/images/left/ev_inf/000400.png', cv2.IMREAD_GRAYSCALE)
# gray_image = cv2.imread('../dsec_dataset_zurich_city_12_a/images/left/ev_inf/000100.png', cv2.IMREAD_GRAYSCALE)
gray_image = cv2.imread('../dsec_dataset_zurich_city_09_b/images/left/ev_inf/000100.png', cv2.IMREAD_GRAYSCALE)

# Loop over various values of gamma
gamma_values = []
metric_values = []
contour_sums = []
step_size = 8

gamma_range1 = np.arange(0.01, 1.5, 0.01)
gamma_range2 = np.arange(2, 21, 2)
gamma_range = np.concatenate((gamma_range1, gamma_range2))

for gamma in gamma_range:
    adjusted_image = adjust_gamma(gray_image, gamma=gamma)
    cv2.imwrite(f'./stacked_events/adjusted_image_gamma_{gamma}.png', adjusted_image)

    gamma_values.append(gamma)

    comparison_metric = compare_images(voxel_grid_event, adjusted_image, step_size=step_size, exclude_coords=exclude_coords)
    metric_values.append(comparison_metric)

    # Compute the sum of contours
    contour_sum = sum_of_contours(adjusted_image)
    contour_sums.append(contour_sum)

    print(f"Gamma: {gamma}, Comparison Metric (Euclidean Distance Mean): {comparison_metric}")

# Find the gamma value that minimizes the comparison metric
best_gamma = gamma_values[np.argmin(metric_values)]
print(f"Best Gamma Value for Metric: {best_gamma}, Contour Sum: {contour_sums[np.argmin(metric_values)]}")

# Find the gamma value that maximizes the sum of contours
best_gamma_contours = gamma_values[np.argmax(contour_sums)]
print(f"Best Gamma Value for Sum of Contours: {best_gamma_contours}, Contour Sum: {np.max(contour_sums)}")

# Plot the cosine distance metric against gamma values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(gamma_values, metric_values, marker='o')
plt.xlabel('Gamma Value, Best Gamma: {:.2f}'.format(best_gamma))
plt.ylabel('SIFT Descriptor Euclidean Distance Mean')
plt.title('SIFT Descriptor Euclidean Distance Mean vs Gamma Value')
plt.grid(True)

# Plot the sum of contours against gamma values
plt.subplot(1, 2, 2)
plt.plot(gamma_values, contour_sums, marker='o', color='r')
plt.xlabel('Gamma Value, Best Gamma: {:.2f}'.format(best_gamma_contours))
plt.ylabel('Sum of Contours')
plt.title('Sum of Contours vs Gamma Value')
plt.grid(True)

plt.tight_layout()
plt.show()

cv2.destroyAllWindows()



