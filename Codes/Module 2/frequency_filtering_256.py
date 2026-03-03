import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# -------------------------------------------------
# Path Configuration
# -------------------------------------------------
folder_path = r"C:\Saintgits\Courses\Computer Vision\Codes\Module 2"
image_path = os.path.join(folder_path, "cameraman_256x256.tif")

# -------------------------------------------------
# 1. Load Image
# -------------------------------------------------
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Image not found!")

image = image.astype(float) / 255.0
rows, cols = image.shape
crow, ccol = rows // 2 , cols // 2

# -------------------------------------------------
# 2. Compute Fourier Transform
# -------------------------------------------------
F = np.fft.fft2(image)
F_shifted = np.fft.fftshift(F)

# -------------------------------------------------
# 3. Create Ideal Low Pass Filter (ILPF)
# -------------------------------------------------
D0 = 10   # Cutoff frequency

mask_LP = np.zeros((rows, cols))
for u in range(rows):
    for v in range(cols):
        D = np.sqrt((u - crow)**2 + (v - ccol)**2)
        if D <= D0:
            mask_LP[u, v] = 1

# -------------------------------------------------
# 4. Create Ideal High Pass Filter (IHPF)
# -------------------------------------------------
mask_HP = 1 - mask_LP

# -------------------------------------------------
# 5. Apply Filters
# -------------------------------------------------
F_LP = F_shifted * mask_LP
F_HP = F_shifted * mask_HP

# -------------------------------------------------
# 6. Inverse Transform
# -------------------------------------------------
img_LP = np.fft.ifft2(np.fft.ifftshift(F_LP))
img_HP = np.fft.ifft2(np.fft.ifftshift(F_HP))

img_LP = np.abs(img_LP)
img_HP = np.abs(img_HP)

# -------------------------------------------------
# 7. Display Results
# -------------------------------------------------
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(np.log(1 + np.abs(F_shifted)), cmap='gray')
plt.title("Magnitude Spectrum")
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(img_LP, cmap='gray')
plt.title("Low Pass Filtered Image")
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(img_HP, cmap='gray')
plt.title("High Pass Filtered Image")
plt.axis('off')

plt.tight_layout()
plt.show()