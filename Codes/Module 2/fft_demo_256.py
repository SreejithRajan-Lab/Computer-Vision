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

image = image.astype(float)

# -------------------------------------------------
# 2. Compute 2D FFT
# -------------------------------------------------
F = np.fft.fft2(image)

# Shift zero frequency to center
F_shifted = np.fft.fftshift(F)

# -------------------------------------------------
# 3. Compute Magnitude and Phase
# -------------------------------------------------
magnitude = np.abs(F_shifted)
phase = np.angle(F_shifted)

log_magnitude = np.log(1 + magnitude)

# -------------------------------------------------
# 4. Inverse FFT (Reconstruction)
# -------------------------------------------------
F_inverse_shift = np.fft.ifftshift(F_shifted)
reconstructed = np.fft.ifft2(F_inverse_shift)
reconstructed = np.abs(reconstructed)

# -------------------------------------------------
# 5. Display Results
# -------------------------------------------------
plt.figure(figsize=(12,10))

plt.subplot(2,2,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(log_magnitude, cmap='gray')
plt.title("FFT Magnitude Spectrum")
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(phase, cmap='gray')
plt.title("FFT Phase Spectrum")
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(reconstructed, cmap='gray')
plt.title("Reconstructed Image (IFFT)")
plt.axis('off')

plt.tight_layout()
plt.show()

print("✔ FFT and IFFT completed successfully.")