import cv2 as cv
import numpy as np

image = cv.imread('../pic/9.jpg')
img = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)

ret, thresh1 = cv.threshold(img, 120, 255, cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(img, 120, 255, cv.THRESH_BINARY_INV)
ret, thresh3 = cv.threshold(img, 120, 255, cv.THRESH_TRUNC)
ret, thresh4 = cv.threshold(img, 120, 255, cv.THRESH_TOZERO)
ret, thresh5 = cv.threshold(img, 120, 255, cv.THRESH_TOZERO_INV)

stacked_img1 = np.hstack((thresh1, thresh2))
stacked_img2 = np.hstack((thresh4, thresh5))

cv.imshow('Binary Threshold - Binary Threshold Inverted', stacked_img1)
# cv.imshow('Binary Threshold Inverted', thresh2)
cv.imshow('Truncated Threshold', thresh3)
cv.imshow('Set to 0 - Set to 0 Inverted', stacked_img2)
# cv.imshow('Set to 0 Inverted', thresh5)

cv.waitKey(0)
cv.destroyAllWindows()
