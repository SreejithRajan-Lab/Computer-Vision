"""
Experiment: Image Smoothing

Objective:
1. Apply Mean Filter
2. Apply Gaussian Filter
3. Compare with Original Image
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Step 1: Load Image
# -------------------------------------------------

img = cv2.imread("cameraman_256x256.tif", 0)

if img is None:
    print("Error: Image not found.")
    exit()

# -------------------------------------------------
# Step 2: Apply Mean Filter (3x3)
# -------------------------------------------------
# Each pixel replaced by average of 3x3 neighborhood

mean_filtered = cv2.blur(img, (3,3))

# -------------------------------------------------
# Step 3: Apply Gaussian Filter (5x5, sigma=1)
# -------------------------------------------------
# Weighted smoothing (center pixel has more weight)

gaussian_filtered = cv2.GaussianBlur(img, (5,5), 1)

# -------------------------------------------------
# Step 4: Display Results
# -------------------------------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(mean_filtered, cmap='gray')
plt.title("Mean Filter (3x3)")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(gaussian_filtered, cmap='gray')
plt.title("Gaussian Filter (5x5, σ=1)")
plt.axis('off')

plt.tight_layout()
plt.show()
