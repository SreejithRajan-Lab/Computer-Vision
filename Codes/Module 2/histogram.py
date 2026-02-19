"""
Experiment 1: Histogram Analysis of 256x256 Cameraman Image
Objective:
1. Load grayscale TIFF image
2. Verify image properties
3. Display image
4. Plot histogram
5. Compute statistical measures (Mean, Std Dev)
"""

# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Step 1: Load the Image (Grayscale Mode)
# ------------------------------------------------------------

# Ensure the TIFF file is in the same directory as this script
image_path = "cameraman.tif"

# Load image in grayscale (0 ensures single channel)
img = cv2.imread(image_path, 0)

# Check if image loaded successfully
if img is None:
    print("Error: Image not found. Check file path.")
    exit()

# ------------------------------------------------------------
# Step 2: Verify Image Properties
# ------------------------------------------------------------

print("----- Image Properties -----")
print("Shape (Rows, Columns):", img.shape)
print("Data Type:", img.dtype)
print("Minimum Intensity:", np.min(img))
print("Maximum Intensity:", np.max(img))

# ------------------------------------------------------------
# Step 3: Compute Statistical Parameters
# ------------------------------------------------------------

mean_intensity = np.mean(img)
std_intensity = np.std(img)
variance = np.var(img)

print("\n----- Statistical Analysis -----")
print("Mean Intensity:", round(mean_intensity, 2))
print("Standard Deviation:", round(std_intensity, 2))
print("Variance:", round(variance, 2))

# Interpretation:
# Mean  -> Overall brightness of image
# Std   -> Measure of contrast (higher = more contrast)

# ------------------------------------------------------------
# Step 4: Display Image and Histogram
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

# Display Original Image
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Cameraman Image")
plt.axis('off')

# Plot Histogram
plt.subplot(1, 2, 2)
plt.hist(img.ravel(), bins=256, range=(0, 256))
plt.title("Histogram of Pixel Intensities")
plt.xlabel("Intensity Value (0-255)")
plt.ylabel("Number of Pixels")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# End of Experiment
# ------------------------------------------------------------

