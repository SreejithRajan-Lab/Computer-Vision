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
    raise FileNotFoundError("Image not found. Check file name/path!")

image = image.astype(float) / 255.0   # Normalize

# -------------------------------------------------
# 2. Continuous Fourier Transform Approximation
#    (Using 2D FFT)
# -------------------------------------------------
F = np.fft.fft2(image)
F_shifted = np.fft.fftshift(F)

# Magnitude Spectrum
magnitude = np.abs(F_shifted)
log_magnitude = np.log(1 + magnitude)

# -------------------------------------------------
# 3. Save Output Image
# -------------------------------------------------
output_path = os.path.join(folder_path, "cft_magnitude_256.png")

cv2.imwrite(output_path,
            (log_magnitude / log_magnitude.max() * 255).astype(np.uint8))

# -------------------------------------------------
# 4. Display Results
# -------------------------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image (256x256)")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(log_magnitude, cmap='gray')
plt.title("Continuous FT Approximation (Log Spectrum)")
plt.axis('off')

plt.tight_layout()
plt.show()

print("✔ Fourier Transform completed successfully.")
print("✔ Output saved at:", output_path)