import cv2 as cv
import numpy as np

# Load image and convert to grayscale
img = cv.imread('../pic/10.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv.Canny(gray, 50, 500, apertureSize=3)

# Apply Hough transform
lines = cv.HoughLines(edges, rho=1, theta=np.pi/180, threshold=160)

# Draw detected lines on the original image
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the result
cv.imshow('edges', edges)
cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()

