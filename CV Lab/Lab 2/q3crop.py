import cv2 as cv

img = cv.imread("../building.jpg", 0)

resized_img = cv.resize(img, (0,0), fx = 0.5, fy = 0.5)
cv.imshow('resized', resized_img)

cropped = img[50:200, 200:400]
cv.imshow("Cropped", cropped)

cv.waitKey(0)