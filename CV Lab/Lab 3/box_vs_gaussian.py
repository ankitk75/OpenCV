import cv2 as cv

img = cv.imread('Resources/3.jpg')

box = cv.boxFilter(img, -1, (5, 5))
gaussian = cv.GaussianBlur(img, (5, 5), 0)

cv.imshow('Box', box)
cv.imshow('Gaussian', gaussian)
cv.waitKey(0)