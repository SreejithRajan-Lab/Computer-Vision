import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("cameraman_256x256.tif", 0)

I_min = np.min(img)
I_max = np.max(img)

contrast = ((img - I_min) / (I_max - I_min) * 255).astype(np.uint8)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(contrast, cmap='gray')
plt.title("Contrast Enhanced")

plt.show()
