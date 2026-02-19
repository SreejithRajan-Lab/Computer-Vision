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

kernel = np.ones((3,3), np.uint8)

min_filter = cv2.erode(img, kernel)
max_filter = cv2.dilate(img, kernel)

plt.subplot(1,2,1)
plt.imshow(min_filter, cmap='gray')
plt.title("Min Filter")

plt.subplot(1,2,2)
plt.imshow(max_filter, cmap='gray')
plt.title("Max Filter")

plt.show()
