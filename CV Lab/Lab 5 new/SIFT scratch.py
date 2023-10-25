import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

original_img = cv.imread('eiffel_1.jpg', 0)

num_octaves = 3
levels_per_octave = 4

gaussian_pyramid = []

# ///////////////////////////////////////////////////////////////////

for octave in range(num_octaves):
    octave_images = []

    for level in range(levels_per_octave):
        if level == 0:
            blurred_img = cv.GaussianBlur(original_img, (7, 7), 0)
        else:
            blurred_img = cv.GaussianBlur(octave_images[-1], (7, 7), 0)
        octave_images.append(blurred_img)

    # Reduce the scale of the original image by half for the next level
    original_img = cv.pyrDown(original_img)
    gaussian_pyramid.append(octave_images)


fig, axs = plt.subplots(num_octaves, levels_per_octave, figsize=(15, 10))
# fig, axs = plt.subplots(num_octaves, levels_per_octave)
for octave in range(num_octaves):
    for level in range(levels_per_octave):
        img = gaussian_pyramid[octave][level]
        axs[octave, level].imshow(img, cmap='gray')
        axs[octave, level].axis('on')
        axs[octave, level].set_title(f'Octave {octave + 1}, Level {level + 1}')

plt.tight_layout()
plt.show()



# Create a list to store the DoG images
dog_pyramid = []

for octave in range(num_octaves):
    dog_images = []
    for level in range(levels_per_octave - 1):
        # Subtract adjacent blurred images to create the DoG image
        dog = gaussian_pyramid[octave][level] - gaussian_pyramid[octave][level + 1]
        dog_images.append(dog)
    dog_pyramid.append(dog_images)

# Visualize the DoG pyramid with scale using Matplotlib
fig, axs = plt.subplots(num_octaves, levels_per_octave - 1, figsize=(15, 10))

for octave in range(num_octaves):
    for level in range(levels_per_octave - 1):
        img = dog_pyramid[octave][level]
        axs[octave, level].imshow(img, cmap='gray')
        # axs[octave, level].axis('on')
        axs[octave, level].set_title(f'Octave {octave + 1}, Level {level + 1}')

plt.tight_layout()
plt.show()

# /////////////////////////////////////////////////////////////////////////////////

# Threshold for keypoint selection
threshold = 0.01  # Adjust this threshold as needed

# List to store detected keypoints
# keypoints = []
keypoints_list = [[] for _ in range(num_octaves * (levels_per_octave - 1))]

# ...

# Loop through each level of the pyramid
for octave in range(num_octaves):
    for level in range(levels_per_octave):  # Include all levels
        current_img = dog_pyramid[octave][level]

        # Skip the first and last levels as they don't have keypoints
        if level == 0 or level == levels_per_octave - 1:
            continue

        for i in range(1, current_img.shape[0] - 1):
            for j in range(1, current_img.shape[1] - 1):
                pixel_value = current_img[i, j]

                # Check if the pixel value is greater or smaller than its 26 neighbors
                if (pixel_value > threshold and
                        pixel_value == np.max([
                            current_img[i - 1:i + 2, j - 1:j + 2],
                            dog_pyramid[octave][level + 1][i - 1:i + 2, j - 1:j + 2],
                            dog_pyramid[octave][level - 1][i - 1:i + 2, j - 1:j + 2]
                        ]) or
                        pixel_value < -threshold and
                        pixel_value == np.min([
                            current_img[i - 1:i + 2, j - 1:j + 2],
                            dog_pyramid[octave][level + 1][i - 1:i + 2, j - 1:j + 2],
                            dog_pyramid[octave][level - 1][i - 1:i + 2, j - 1:j + 2]
                        ])):
                    keypoints_list[octave * (levels_per_octave - 2) + level - 1].append(
                        (j, i))  # Store the (x, y) coordinate of the keypoint

# Visualize the keypoints on respective images
plt.figure(figsize=(15, 10))

for octave in range(num_octaves):
    for level in range(1, levels_per_octave - 1):  # Avoid the first and last levels
        idx = octave * (levels_per_octave - 2) + level - 1
        plt.subplot(num_octaves, levels_per_octave - 2, idx + 1)
        img = dog_pyramid[octave][level]
        plt.imshow(img, cmap='gray')
        plt.scatter(*zip(*keypoints_list[idx]), s=1, c='r', marker='o')
        plt.title(f'Octave {octave + 1}, Level {level}')
        plt.axis('off')

plt.tight_layout()
plt.show()

# Loop through each level of the pyramid
for octave_images in dog_pyramid:
    for level in range(1, len(octave_images) - 1):  # Avoid first and last levels
        current_img = octave_images[level]
        above_img = octave_images[level + 1]
        below_img = octave_images[level - 1]

        for i in range(1, current_img.shape[0] - 1):
            for j in range(1, current_img.shape[1] - 1):
                pixel_value = current_img[i, j]

                # Check if the pixel value is greater or smaller than its 26 neighbors
                if (pixel_value > threshold and
                        pixel_value == np.max([
                            current_img[i - 1:i + 2, j - 1:j + 2],
                            above_img[i - 1:i + 2, j - 1:j + 2],
                            below_img[i - 1:i + 2, j - 1:j + 2]
                        ]) or
                        pixel_value < -threshold and
                        pixel_value == np.min([
                            current_img[i - 1:i + 2, j - 1:j + 2],
                            above_img[i - 1:i + 2, j - 1:j + 2],
                            below_img[i - 1:i + 2, j - 1:j + 2]
                        ])):
                    keypoints.append((j, i))  # Store the (x, y) coordinate of the keypoint

# Visualize the keypoints on an example image (replace 'example_img' with your image)
example_img = cv.imread('eiffel_1.jpg', 0)
plt.figure(figsize=(8, 8))
plt.imshow(example_img, cmap='gray')
plt.scatter(*zip(*keypoints), s=1, c='r', marker='o')
plt.title('Detected Keypoints')
# plt.axis('on')
plt.show()


# Thresholds for keypoint selection
contrast_threshold = 0.03  # Adjust as needed
curvature_threshold = 10  # Adjust as needed
edge_response_threshold = 20  # Adjust as needed
#
# List to store selected keypoints
selected_keypoints = []

# Loop through each level of the pyramid
for octave_images in dog_pyramid:
    for level in range(1, len(octave_images) - 1):  # Avoid first and last levels
        current_img = octave_images[level]

        for x, y in keypoints:  # Assuming you have a list of detected keypoints
            # Check contrast
            pixel_value = current_img[y, x]
            if abs(pixel_value) < contrast_threshold:
                continue

            # Calculate the Hessian matrix
            Dxx = current_img[y, x + 1] + current_img[y, x - 1] - 2 * pixel_value
            Dyy = current_img[y + 1, x] + current_img[y - 1, x] - 2 * pixel_value
            Dxy = (current_img[y + 1, x + 1] - current_img[y + 1, x - 1] -
                   current_img[y - 1, x + 1] + current_img[y - 1, x - 1]) / 4.0

            # Calculate the curvature
            trace = Dxx + Dyy
            determinant = Dxx * Dyy - Dxy * Dxy
            curvature = trace * trace / determinant if determinant != 0 else float('inf')

            # Check curvature and edge response
            if curvature < curvature_threshold:
                edge_response = (Dxx + Dyy) * (Dxx + Dyy) / determinant
                if edge_response < edge_response_threshold:
                    selected_keypoints.append((x, y))

# Now, selected_keypoints contains the robust keypoints that passed the contrast and edge checks





# Thresholds for keypoint selection
contrast_threshold = 0.03  # Adjust as needed
curvature_threshold = 10  # Adjust as needed
edge_response_threshold = 20  # Adjust as needed

# List to store selected keypoints
selected_keypoints = []

# Loop through each level of the pyramid
for octave_images in dog_pyramid:
    for level in range(1, len(octave_images) - 1):  # Avoid first and last levels
        current_img = octave_images[level]

        for x, y in keypoints:  # Assuming you have a list of detected keypoints
            # Check contrast
            pixel_value = current_img[y, x]
            if abs(pixel_value) < contrast_threshold:
                continue

            # Calculate the Hessian matrix
            Dxx = current_img[y, x + 1] + current_img[y, x - 1] - 2 * pixel_value
            Dyy = current_img[y + 1, x] + current_img[y - 1, x] - 2 * pixel_value
            Dxy = (current_img[y + 1, x + 1] - current_img[y + 1, x - 1] -
                   current_img[y - 1, x + 1] + current_img[y - 1, x - 1]) / 4.0

            # Calculate the curvature
            trace = Dxx + Dyy
            determinant = Dxx * Dyy - Dxy * Dxy
            curvature = trace * trace / determinant if determinant != 0 else float('inf')

            # Check curvature and edge response
            if curvature < curvature_threshold:
                edge_response = (Dxx + Dyy) * (Dxx + Dyy) / determinant
                if edge_response < edge_response_threshold:
                    selected_keypoints.append((x, y))

# Convert the example image to color for visualization
example_img_color = cv.cvtColor(example_img, cv.COLOR_GRAY2BGR)

# Draw circles around selected keypoints
for x, y in selected_keypoints:
    cv.circle(example_img_color, (x, y), 5, (0, 0, 255), -1)  # Draw a red circle around the keypoint

# Display the image with selected keypoints
plt.figure(figsize=(8, 8))
plt.imshow(cv.cvtColor(example_img_color, cv.COLOR_BGR2RGB))
plt.title('Image with Selected Keypoints')
plt.axis('off')
plt.show()






