import cv2 as cv
import numpy as np

image = cv.imread('../pic/10.jpg')

height, width = image.shape[:2]

# Reshape the image to a 2D array of pixels and 3 color channels
pixels = image.reshape((-1, 3))

# Convert the data type to float32 for K-means clustering
pixels = np.float32(pixels)

# Define the number of clusters (segments) you want
num_clusters = 5

# Define the criteria for K-means algorithm
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Apply K-means clustering
_, labels, centers = cv.kmeans(pixels, num_clusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# Convert the cluster centers back to uint8
centers = np.uint8(centers)

# Map each pixel to its corresponding cluster center
segmented_image = centers[labels.flatten()]

# Reshape the segmented image back to its original shape
segmented_image = segmented_image.reshape(image.shape)

# Display the original image and the segmented image

stacked_img = np.hstack((image, segmented_image))

cv.imshow('Original - Segmented', stacked_img)
# cv.imshow('Segmented Image', segmented_image)
cv.waitKey(0)
cv.destroyAllWindows()
