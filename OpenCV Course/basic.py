import cv2 as cv

img = cv.imread('/Users/ankitkumar/Desktop/College Stuff/3rd Year/OpenCV/portrait.JPG')
cv.imshow('Evening', img)

# Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("Gray", gray)

# Blur
blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
cv.imshow("Blur", blur)

# Edge Cascade
canny = cv.Canny(blur , 125, 175)
cv.imshow("Canny", canny)

# Dilating the image
dilated = cv.dilate(canny, (7,7), iterations = 3)
cv.imshow("Dilated", dilated)

# Eroding
erode = cv.erode(dilated, (7,7  ), iterations=3)
cv.imshow('Erode', erode)

# Resize
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow("Resized", resized)

# Cropping
cropped = img[50:200, 200:400]
cv.imshow("Croopped", cropped)

cv.waitKey(0)