import cv2 as cv

img = cv.imread('/Users/ankitkumar/Desktop/College Stuff/3rd Year/OpenCV/evening.JPG')

cv.imshow('Evening', img)

color = img[100, 200]

blue = int(color[0])
green = int(color[1])
red = int(color[2])

print(f'Blue: {blue}\nGreen: {green}\nRed: {red}')

height, width = img.shape[:2]

print("Image width:", width)
print("Image height:", height)

resized_img = cv.resize(img,(0,0), fx = 0.5, fy = 0.5)
cv.imshow('Evening2', resized_img)

rot_img = cv.rotate(img, )

cv.waitKey(0)
# Reading videos

# capture = cv.VideoCapture('/Users/ankitkumar/Desktop/College Stuff/3rd Year/OpenCV/pexels_video.mp4')

# while True:
#     isTrue, frame = capture.read()
#
#     cv.imshow("Video", frame)
#
#     if cv.waitKey(20) & 0xFF ==  ord('d'):
#         break
#
# # At the end of the video it gives an -215 assertion failed error  because it couldn't find any frame after the video ended
#
# capture.release()
cv.destroyAllWindows()

