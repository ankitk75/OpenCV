import cv2
import numpy as np
import matplotlib.pyplot as plt

input_img = cv2.imread('../building.jpg', cv2.IMREAD_GRAYSCALE)
reference_img = cv2.imread('../sand.jpg', cv2.IMREAD_GRAYSCALE)

hist_input, _ = np.histogram(input_img.flatten(), bins=256, range=[0, 256])
hist_reference, _ = np.histogram(reference_img.flatten(), bins=256, range=[0, 256])

cdf_input = hist_input.cumsum()
cdf_reference = hist_reference.cumsum()

cdf_input_normalized = (cdf_input - cdf_input.min()) * 255 / (cdf_input.max() - cdf_input.min())
cdf_reference_normalized = (cdf_reference - cdf_reference.min()) * 255 / (cdf_reference.max() - cdf_reference.min())

lut = np.interp(cdf_input_normalized, cdf_reference_normalized, np.arange(256))

output_img = cv2.LUT(input_img, lut.astype(np.uint8))

plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.title('Input Image')
plt.imshow(input_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Reference Image')
plt.imshow(reference_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Output Image')
plt.imshow(output_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Histogram of Input Image')
plt.hist(input_img.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.xlim([0, 256])

plt.subplot(2, 3, 5)
plt.title('Histogram of Reference Image')
plt.hist(reference_img.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.xlim([0, 256])

plt.subplot(2, 3, 6)
plt.title('Histogram of Output Image')
plt.hist(output_img.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.xlim([0, 256])

plt.tight_layout()
plt.show()