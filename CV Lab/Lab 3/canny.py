import cv2 as cv

img = cv.imread('Resources/3.jpg')

canny = cv.Canny(img, 100, 200)

cv.imshow('Original', img)
cv.imshow('Canny', canny)
cv.waitKey(0)