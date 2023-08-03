import cv2 as cv

img = cv.imread("church.jpg")
cv.imshow('Chuch', img)

#Translation
def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimension = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimension)

# -x --> Left
# -y --> Up
# x --> Right
# y --> Down

translated = translate(img, 100, 100)
cv.imshow("Translated", translated)


# Rotation

def rotate(img, angle, rotPoint = None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width // 2, height // 2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimension = (width, height)

    return cv.warpAffine(img, rotMat, dimension)


rotated = rotate(img, -45)
cv.imshow("Rotated", rotated )


rr = rotate(rotated, -45)
cv.imshow('Rotated_2', rr)


# Resizing
resized = cv. resize(img, (500,500) , interpolation=cv. INTER_CUBIC)
cv. imshow ('Resized', resized)

# Flippina
flip = cv. flip(img, 1)
cv. imshow ('Flip', flip)

# Cropping
cropped = img[200:400, 300:400]
cv. imshow('Cropped', cropped)



cv.waitKey(0 )
