import cv2 as cv
import numpy as np

threshold = 0.6
img = cv.imread('../pic/7.jpg')
dx = cv.Sobel(img, cv.CV_64F, 1, 0)
dy = cv.Sobel(img, cv.CV_64F, 0, 1)

dx2 = np.square(dx)
dy2 = np.square(dy)
dxdy = dx*dy

g_dx2 = cv.GaussianBlur(dx2,(3, 3), 0)
g_dy2 = cv.GaussianBlur(dy2,(3, 3), 0)
g_dxdy = cv.GaussianBlur(dxdy,(3, 3), 0)

harris = g_dx2 * g_dy2 - np.square(g_dxdy) - 0.12 * np.square(g_dx2 + g_dy2)
cv.normalize(harris, harris, 0, 1, cv.NORM_MINMAX)

# find all points above threshold (nonmax supression line)
loc = np.where(harris >= threshold)
# drawing filtered points
for pt in zip(*loc[::-1]):
    cv.circle(img, pt, 3, (0, 0, 255), -1)

cv.waitKey(0)
# return img_cpy,g_dx2,g_dy2,dx,dy,loc


























# img = cv.imread('../pic/7.jpg')
# operatedImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# operatedImage = np.float32(operatedImage)
# dest = cv.cornerHarris(operatedImage, 2, 5, 0.07)
# dest = cv.dilate(dest, None)
# img[dest > 0.01 * dest.max()] = [0, 0, 255]
# cv.imshow('Image with Borders', img)
# if cv.waitKey(0) & 0xff == 27:
#     cv.destroyAllWindows()