"""
Experiment: Median Filter for Noise Removal

Objective:
1. Add Salt & Pepper noise
2. Apply Median filter
3. Compare results
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
# Step 2: Add Salt & Pepper Noise
# -------------------------------------------------

noisy = img.copy()

prob = 0.02  # noise probability

# Salt noise
salt = np.random.rand(*img.shape) < prob
noisy[salt] = 255

# Pepper noise
pepper = np.random.rand(*img.shape) < prob
noisy[pepper] = 0

# -------------------------------------------------
# Step 3: Apply Median Filter (3x3)
# -------------------------------------------------

median_filtered = cv2.medianBlur(noisy, 3)

# -------------------------------------------------
# Step 4: Display Results
# -------------------------------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(noisy, cmap='gray')
plt.title("Salt & Pepper Noise")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(median_filtered, cmap='gray')
plt.title("Median Filter Result")
plt.axis('off')

plt.tight_layout()
plt.show()
