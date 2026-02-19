"""
Experiment: Contrast Normalisation (Standardization)

Objective:
1. Normalize image intensity using:
       I_norm = (I - mean) / std
2. Display normalized image properly
3. Verify new mean and std
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
# Step 2: Compute Mean and Standard Deviation
# -------------------------------------------------

mean = np.mean(img)
std = np.std(img)

print("Original Mean:", round(mean, 2))
print("Original Std Dev:", round(std, 2))

# -------------------------------------------------
# Step 3: Apply Contrast Normalisation
# -------------------------------------------------

normalized = (img - mean) / std

print("After Normalisation:")
print("New Mean:", round(np.mean(normalized), 5))
print("New Std Dev:", round(np.std(normalized), 5))

# -------------------------------------------------
# Step 4: Rescale for Display (0–255)
# -------------------------------------------------
# Because normalized values contain negatives

norm_min = np.min(normalized)
norm_max = np.max(normalized)

normalized_display = ((normalized - norm_min) / 
                      (norm_max - norm_min) * 255)

normalized_display = normalized_display.astype(np.uint8)

# -------------------------------------------------
# Step 5: Display Results
# -------------------------------------------------

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(normalized_display, cmap='gray')
plt.title("Contrast Normalised (Rescaled for Display)")
plt.axis('off')

plt.show()
