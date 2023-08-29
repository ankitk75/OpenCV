import numpy as np
import cv2 as cv

img = cv.imread('Resources/4.jpg', 0)
lap = cv.Laplacian(img, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('OG', img)
cv.imshow('Laplacian Gradient', lap)

# sobelx = cv.Sobel(img, cv.CV_64F, 1, 0)
# sobely = cv.Sobel(img, cv.CV_64F, 0, 1)
# sobelx = np.uint8(np.absolute(sobelx))
# sobely = np.uint8(np.absolute(sobely))
# sobelcombined = cv.bitwise_or(sobelx, sobely)
# cv.imshow('OG', img)
# cv.imshow('Sobel X', sobelx)
# cv.imshow('Sobel Y', sobely)
# cv.imshow('Sobel Combined', sobelcombined)
cv.waitKey(0)











# img = cv.imread('Resources/4.jpg', 0)
# # img = cv.resize(img, dsize=None, fx=0.5, fy=0.5)
# lap = cv.Laplacian(img, cv.CV_64F)
# lap = np.uint8(np.absolute(lap))
# cv.imshow('OG', img)
# cv.imshow('Laplacian Gradient', lap)
#
# cv.waitKey(0)

# import numpy as np
# import cv2 as cv


# def segment_image(image_path):
#     # Read the image
#     image = cv.imread(image_path)
#
#     # Convert image from BGR to HSV color space
#     hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
#
#     # Define the range of colors you want to segment (for demonstration, let's say green)
#     lower_green = np.array([40, 40, 40])  # Lower range of green color in HSV
#     upper_green = np.array([80, 255, 255])  # Upper range of green color in HSV
#
#     # Create a binary mask where green regions are white and the rest is black
#     mask = cv.inRange(hsv_image, lower_green, upper_green)
#
#     # Apply the mask to the original image to get the segmented regions
#     segmented_image = cv.bitwise_and(image, image, mask=mask)
#
#     # Display the original image and segmented image
#     cv.imshow('Original Image', image)
#     cv.imshow('Segmented Image', segmented_image)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     image_path = 'Resources/4.jpg'
#     segment_image(image_path)
