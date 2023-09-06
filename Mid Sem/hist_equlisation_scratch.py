import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image):
    hist = cv.calcHist([image], [0], None, [256], [0, 256])

    hist /= hist.sum()
    cdf = np.cumsum(hist)
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    #For example, if image[0, 0] is 100 and cdf_normalized[100] is 150, then equalized_image[0, 0] will become 150.
    equalized_image = cdf_normalized[image]
    hist1 = cv.calcHist([equalized_image], [0], None, [256], [0, 256])

    return equalized_image, hist, hist1

image = cv.imread('../../Resources/1.jpg', 0)
equalized_image, hist, hist1 = histogram_equalization(image)

plt.subplot(2, 2, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')

plt.subplot(2, 2, 3)
plt.plot(hist)
plt.title('og histogram')

plt.subplot(2, 2, 4)
plt.plot(hist1)
plt.title('eq histogram')

plt.show()
