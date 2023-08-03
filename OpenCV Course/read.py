import cv2 as cv

# img = cv.imread('evening.JPG')

# cv.imshow('Evening', img)

# Reading videos

capture = cv.VideoCapture('pexels_video.mp4')

while True:
    isTrue, frame = capture.read()

    cv.imshow("Video", frame)

    if cv.waitKey(20) & 0xFF ==  ord('d'):
        break

# At the end of the video it gives an -215 assertion failed error  because it couldn't find any frame after the video ended

capture.release()
cv.destroyAllWindows()

# cv.waitKey(0)
