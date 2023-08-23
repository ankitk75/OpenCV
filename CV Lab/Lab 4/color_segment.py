import cv2 as cv
import numpy as np

# Load an image
image_path = '../pic/7.jpg'
image = cv.imread(image_path)

# Convert the image from BGR to HSV color space
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Define the lower and upper bounds of the color you want to segment
lower_color = np.array([20, 50, 50])  # Replace with your desired lower bound
upper_color = np.array([40, 255, 255])  # Replace with your desired upper bound

# Create a mask using the specified color range
mask = cv.inRange(hsv_image, lower_color, upper_color)

# Apply the mask to the original image
segmented_image = cv.bitwise_and(image, image, mask=mask)

# Display the original image and the segmented image
cv.imshow('Original Image', image)
cv.imshow('Segmented Image', segmented_image)
cv.waitKey(0)
cv.destroyAllWindows()
