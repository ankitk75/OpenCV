import cv2 as cv

img = cv.imread("/Users/ankitkumar/Desktop/College Stuff/3rd Year/OpenCV/blights.jpg", 0)
cv.imshow('Chuch', img)

canny = cv. Canny (img, 125, 175)
cv. imshow( 'Canny Edges', canny)


cv.waitKey(0)