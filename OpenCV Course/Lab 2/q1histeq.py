import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("../park.jpg")
# cv.imshow('img', img)

gray = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
gray_hist = cv.calcHist([gray], [0], None, [256], [0,256])
plt.subplot(2,2,1)
plt.plot(gray_hist)
plt.xlim(0,256)
plt.title("Gray Histogram")
# plt.show()

equ = cv.equalizeHist(gray)
equ_hist = cv.calcHist([equ], [0], None, [256], [0,256])
plt.subplot(2,2,2)
plt.plot(equ_hist)
plt.xlim(0,256)
plt.title("Equ Gray Histogram")
plt.show()

res = np.hstack((gray, equ))

cv.imshow('OG-EQ', res)
cv.waitKey(0)
cv.destroyAllWindows()
