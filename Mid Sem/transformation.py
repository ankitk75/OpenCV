import cv2
import numpy as np

#Log transforms are used to expand dark pixels in an image while compressing higher-level values.

img1 = cv2.imread('../../Resources/1.jpg', 0)

# Apply log transform.
c = 255/(np.log(1 + np.max(img1)))
log_transformed = c * np. log(1 + img1)

log_transformed = np.array(log_transformed, dtype = np.uint8)
cv2.imshow('log_transformed', log_transformed)




# Gamma correction is the process of changing the brightness values of an image by some gamma factor.
img2 = cv2. imread('../../Resources/1.jpg', 0)

# Trying 4 gamma values.
for gamma in [0.1, 0.5, 1.2, 2.2]:
    # Apply gamma correction.
    gamma_corrected = np.array (255 * (img2 / 255) ** gamma, dtype = 'uint8')

cv2.imshow('gamma_transformed', gamma_corrected)
cv2.waitKey(0)




# Contrast stretching is a method that improves the contrast in an image
# by stretching the range of intensity values it contains to span a desired range of values.
# Function to map each intensity level to output intensity level.
def pixelVal (pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((52 - 51)/(r2 - r1)) * (pix - r1) + 51
    else:
        return ((255 - 52)/(255 - r2)) * (pix - r2) + 52

img = cv2.imread( "../../Resources/2.jpg", 0)

# Define parameters.
r1 = 70
s1 = 0
r2 = 140
s2 = 255

# Vectorize the function to apply it to each value in the Numpy array.
pixelVal_vec = np.vectorize (pixelVal)
# Apply contrast stretching.
contrast_stretched = pixelVal_vec(img, r1, 51, r2, s2)

cv2.imshow('original', img)
cv2.imshow('contrast_stretching', contrast_stretched)
