# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Load the image
# img = cv.imread('image.jpg')
#
# # Resize the image
# img1 = cv.resize(img, (0, 0), fx=1, fy=1)
#
# # Convert the image to grayscale
# img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# print(img.shape)
#
# # Parameters for corner detection
# corners = []
# t = 100
# n = 12
#
# # Loop through the image pixels
# for y in range(3, img.shape[0]-3):
#     for x in range(3, img.shape[1]-3):
#         b, d = 0, 0
#         pts1 = [(y-3, x), (y+3, x), (y, x-3), (y, x+3)]
#         pts = [(y-2, x-2), (y+2, x+2), (y-2, x+2), (y+2, x-2),
#                (y-3, x-1), (y-3, x+1), (y+3, x-1), (y+3, x+1),
#                (y-1, x-3), (y+1, x-3), (y-1, x+3), (y+1, x+3)]
#
#         # Initial corner detection
#         for point in pts1:
#             if img[point[0], point[1]] > img[y, x] + t:
#                 b += 1
#             if img[point[0], point[1]] < img[y, x] - t:
#                 d += 1
#
#         # Refining the corners
#         if b <= 2 and d <= 2:
#             continue
#         elif b > 2 or d > 2:
#             for point in pts:
#                 if img[point[0], point[1]] > img[y, x] + t:
#                     b += 1
#                 elif img[point[0], point[1]] < img[y, x] - t:
#                     d += 1
#
#         # Confirming the corners
#         if b >= n or d >= n:
#             corners.append((y, x))
#
# # Output the number of corners detected
# print(len(corners))
#
# # Visualizing the corners on the image
# for y, x in corners:
#     cv.circle(img1, (x, y), 3, (0, 255, 0), -1)
#
# # Display the image with corners
# cv.imshow('image', img1)
# cv.waitKey(0)

import cv2
import numpy as np

# Load and resize the image
img = cv2.imread('eiffel_1.jpg', 0)
img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))  # Note: cv2.resize takes width first, then height

# Parameters for the algorithm
n = 12
t = 75

# Convert grayscale image to BGR for drawing colored circles
corners = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Circle coordinates
circle = [(-3, 0), (-3, 1), (-2, 2), (-1, 3), (0, 3), (1, 3), (2,2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3), (0, -3), (-1, -3), (-2, -2), (-3, -1)]

# Loop through the image pixels
for i in range(3, img.shape[0]-3):
    for j in range(3, img.shape[1]-3):
        center = img[i, j]
        for s in range(16):
            a, b = 0, 0
            above, below = True, True
            for k in range(s, s+n):
                ni, nj = i+circle[k%16][0], j+circle[k%16][1]
                if img[ni, nj] > center+t:
                    a += 1
                    below = False
                elif img[ni, nj] < center-t:
                    b += 1
                    above = False
                else:
                    above = False
                    below = False
            if (above or below) and (a == n or b == n):
                corners = cv2.circle(corners, (j, i), 3, (0, 0, 255), 1)
                break

# FAST Feature Detection
fast = cv2.FastFeatureDetector_create()
kp = fast.detect(img, None)
img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

# Convert the grayscale image to have 3 channels so it can be concatenated and displayed together
img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Concatenate the images horizontally
concatenated_img = cv2.hconcat([img_colored, img2, corners])

# Display the concatenated image
cv2.imshow('Output', concatenated_img)

# Wait for a key event and then close all OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()