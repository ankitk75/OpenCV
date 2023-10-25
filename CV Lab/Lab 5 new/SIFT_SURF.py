import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_transformation(image, transformation_type):
    rows, cols = image.shape

    if transformation_type == 'scale':
        return cv2.resize(image, None, fx=1.5, fy=1.5)

    elif transformation_type == 'rotate':
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        # Swapping rows and cols values for the rotated image
        transformed_image = cv2.warpAffine(image, M, (rows, cols))  # Swapped cols and rows

    elif transformation_type == 'affine':
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)
        transformed_image = cv2.warpAffine(image, M, (cols, rows))

    return transformed_image

def apply_feature_detection(image1, image2, method):
    if method == 'sift':
        feature_detector = cv2.SIFT_create()
    elif method == 'surf':
        feature_detector = cv2.xfeatures2d.SURF_create()

    keypoints1, descriptors1 = feature_detector.detectAndCompute(image1, None)
    keypoints2, descriptors2 = feature_detector.detectAndCompute(image2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.1 * n.distance:
            good_matches.append(m)

    return keypoints1, keypoints2, good_matches


image_path = 'eiffel_1.jpg'
image = cv2.imread(image_path, 0)

if image is None:
    print(f"Error: Unable to load image at path: {image_path}")
    exit()

transformations = ['scale', 'rotate', 'affine']
# methods = ['sift', 'surf']
methods = ['sift']
for transformation in transformations:
    transformed_image = apply_transformation(image, transformation)

    for method in methods:
        keypoints1, keypoints2, good_matches = apply_feature_detection(image, transformed_image, method)
        img_matches = cv2.drawMatches(image, keypoints1, transformed_image, keypoints2, good_matches, None)

        plt.imshow(img_matches)
        plt.title(f'{method.upper()} - {transformation.capitalize()}')
        plt.show()