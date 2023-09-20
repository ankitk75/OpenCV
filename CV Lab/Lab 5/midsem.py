import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('1.jpg')

# height, width = image.shape[:2]
# pixels = image.reshape((-1, 3))
# pixels = np.float32(pixels)
#
# num_clusters = 3
#
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# _, labels, centers = cv.kmeans(pixels, num_clusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
# centers = np.uint8(centers)
# segmented_image = centers[labels.flatten()]
# segmented_image = segmented_image.reshape(image.shape)
# # segmented_image = cv.resize('segmented_image',(0,0), fx=0.5, fy=0.5)

# stacked_img = np.hstack((image, segmented_image))
# cv.imshow('Original - Segmented', stacked_img)

image = cv.medianBlur(image,7)
image = cv.GaussianBlur(image, (11,11), 0)
ret, thresh = cv.threshold(image, 120, 255, cv.THRESH_BINARY)

# kernel = np.ones((3, 3), np.uint8)
# closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE,kernel, iterations = 15)
# bg = cv.dilate(closing, kernel, iterations = 1)
# dist_transform = cv.distanceTransform(closing, cv.DIST_L2, 0)
# reta, fg = cv.threshold(dist_transform, 0.02*dist_transform.max(), 255, 0)
# cv.imshow('image', fg)
# plt.figure(figsize=(8,8))
# plt.imshow(fg,cmap="gray")
# plt.axis('off')
# plt.title("Segmented Image")
# plt.show()

lower_color = np.array([70, 90, 90])
upper_color = np.array([255, 255, 255])

mask = cv.inRange(thresh, lower_color, upper_color)

seg = cv.bitwise_and(thresh, thresh, mask=mask)

cv.imshow('Original Image', image)
cv.imshow('Segmented Image', thresh)
cv.imshow('try', seg)
cv.waitKey(0)
cv.destroyAllWindows()