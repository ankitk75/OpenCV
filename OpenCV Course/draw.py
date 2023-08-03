import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype = 'uint8')
cv.imshow('blank', blank)

# 1.Paint the image a certain colour
#  imshow('Green', blank)
# blank[:] = 0,0,255
# cv.imshow('Red', blank)
# blank[:] = 255,0,0
# cv.imshow('Blue', blank)

# 2.Draw a Rectangle
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), ( 0,255,0), thickness = cv.FILLED)    # -1 for filled
cv.imshow("Rectangle", blank)


# 3.Draw a circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 50, (0,0,255), thickness = -1  )
cv.imshow("Circle", blank)

# 4.Draw a line
cv.line(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (255,255,255), thickness = 3)
cv.imshow("Line", blank )

# 5. Write Text
cv.putText(blank, "Hello", (225,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 1  )
cv.imshow("Text", blank)

cv.waitKey(0)