# import cv2
# import numpy as np
#
# img=cv2.imread('eiffel_1.jpg', 0)
# print(img.shape)
#
# octaves=[]
# dog=[]
#
# #image pyramid for 3 octaves, 4 scales each and calculating difference of Gaussian
# for octave in range(3):
#     gaussians=[]
#     scalediff=[]
#     sigma=2**(octave)*1/2**0.5
#     for scale in range(4):
#         blurred=cv2.GaussianBlur(img, (7,7),sigma)
#         gaussians.append(blurred)
#         print(sigma)
#         sigma=sigma*2**0.5
#         #cv2.imshow('Blurred', blurred)
#         #cv2.waitKey(0)
#         if scale>0:
#             diff=gaussians[scale]-gaussians[scale-1]
#             scalediff.append(diff)
#     octaves.append(gaussians)
#     dog.append(scalediff)
# # for i in range(3):
# #     for j in range(3):
# #         s=dog[i][j]
# #         print(s.shape)
# #         print(s)
# #         cv2.imshow('Diff', s)
# #         cv2.waitKey(0)
#
# n=0
# allkeypoints=np.zeros(img.shape, dtype='uint8')
#
# for octave in range(3):
#     keypoints=np.zeros(img.shape, dtype='uint8')
#     diff=dog[octave][1]
#     for i in range(1, img.shape[0]-1):
#         for j in range(1, img.shape[1]-1):
#             k=1
#             for p in (dog[octave][0][i-1:i+2, j-1:j+2]).reshape(9):
#                 if diff[i,j]<=p:
#                     k=0
#                     break
#             for p in (dog[octave][2][i-1:i+2, j-1:j+2]).reshape(9):
#                 if diff[i,j]<=p:
#                     k=0
#                     break
#             for x in range(3):
#                 for y in range(3):
#                     if (x!=1 and y!=1):
#                         if diff[i,j]<=diff[i+x-1, j+y-1]:
#                             k=0
#                             break
#
#
#             keypoints[i,j]=k*255
#             if k==1:
#                 n+=1
#                 img=cv2.circle(img, (j,i), 3,(0,0,255), 1)
#     cv2.imshow('Octave keypoint', keypoints)
#     cv2.waitKey(0)
#     allkeypoints+=keypoints
# sift = cv2.SIFT_create()
# kp = sift.detect(img,None)
# points=cv2.drawKeypoints(img,kp,img)
# print(len(kp))
# print(n)
# cv2.imshow('Function', points)
# cv2.imshow('All keypoints', allkeypoints)
# cv2.imshow('Original', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np

# Read the grayscale image
img = cv2.imread('eiffel_1.jpg', 0)
print(img.shape)

octaves = []
dog = []

# Image pyramid for 3 octaves, 4 scales each, and calculating the difference of Gaussian
for octave in range(3):
    gaussians = []
    scalediff = []
    sigma = 2 ** (octave) * 1 / 2 ** 0.5
    for scale in range(4):
        blurred = cv2.GaussianBlur(img, (7, 7), sigma)
        gaussians.append(blurred)
        print(sigma)
        sigma = sigma * 2 ** 0.5
        if scale > 0:
            diff = gaussians[scale] - gaussians[scale - 1]
            scalediff.append(diff)
    octaves.append(gaussians)
    dog.append(scalediff)

n = 0
allkeypoints = np.zeros(img.shape, dtype='uint8')

# Loop through each octave to find keypoints
for octave in range(3):
    keypoints = np.zeros(img.shape, dtype='uint8')
    diff = dog[octave][1]
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            k = 1
            for p in (dog[octave][0][i - 1:i + 2, j - 1:j + 2]).reshape(9):
                if diff[i, j] <= p:
                    k = 0
                    break
            for p in (dog[octave][2][i - 1:i + 2, j - 1:j + 2]).reshape(9):
                if diff[i, j] <= p:
                    k = 0
                    break
            for x in range(3):
                for y in range(3):
                    if (x != 1 and y != 1):
                        if diff[i, j] <= diff[i + x - 1, j + y - 1]:
                            k = 0
                            break

            keypoints[i, j] = k * 255
            if k == 1:
                n += 1
                img = cv2.circle(img, (j, i), 3, (0, 0, 255), 1)
    cv2.imshow('Octave keypoint', keypoints)
    cv2.waitKey(0)
    allkeypoints += keypoints

# Create a SIFT detector
sift = cv2.SIFT_create()
kp = sift.detect(img, None)

# Draw SIFT keypoints on the image
points = cv2.drawKeypoints(img, kp, img)
print("Number of SIFT keypoints:", len(kp))
print("Number of custom keypoints:", n)

# Display the images
cv2.imshow('Function', points)
cv2.imshow('All keypoints', allkeypoints)
cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
