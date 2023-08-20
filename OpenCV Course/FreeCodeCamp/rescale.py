import cv2 as cv

img = cv.imread('/Users/ankitkumar/Desktop/College Stuff/3rd Year/OpenCV/evening.JPG')
cv.imshow('Evening', img)



def rescaleFrame(frame, scale = 0.75):
    # Images, Videos snd Live Videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimension = (width, height)

    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)



def changeRes(width, height):
    # Live Videos
    capture.set(3, width)
    capture.set(4, height)



resized_img = rescaleFrame(img, .5)
cv.imshow('resized image', resized_img)

cv.waitKey(0)

# Reading videos

capture = cv.VideoCapture('/Users/ankitkumar/Desktop/College Stuff/3rd Year/OpenCV/pexels_video.mp4')

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame, .5)

    cv.imshow("Video", frame)
    cv.imshow("Video_resized", frame_resized)

    if cv.waitKey(20) & 0xFF ==  ord('d'):
        break

capture.release()
cv.destroyAllWindows()

