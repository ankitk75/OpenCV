import numpy as np
import cv2 as cv

img = cv.imread('Resources/3 .jpg', 0)
filtered = cv.GaussianBlur(img, (5, 5), 0)

img = np.array(img)
filtered = np.array(filtered)
img1 = img - filtered
img_unsharp = img + img1

cv.imshow('original', img)
cv.imshow('filtered', filtered)
cv.imshow('unsharp mask', img1)
cv.imshow('unsharped', img_unsharp)
cv.waitKey(0)


# import cv2 as cv
#
# img = cv.imread('Resources\\ny_ts.jpg')
# img = cv.resize(img, dsize=None, fx=0.5, fy=0.5)
# gaussian = cv.GaussianBlur(img, (11,11), 0)
# unsharp = cv.addWeighted(img, 2.0, gaussian, -1.0, 0)
# cv.imshow('OG', img)
# cv.imshow('Unsharp', unsharp)
#
# cv.waitKey(0)