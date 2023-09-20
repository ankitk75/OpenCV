import cv2 as cv
import numpy as np
def compute_gradient(image):
    # Calculate gradient using Sobel filters
    gradient_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    gradient_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    angle = np.arctan2(gradient_y, gradient_x)

    return magnitude, angle


def quantize_orientation(angles, bins=9):
    # Quantize angles into orientation bins
    angles = np.rad2deg(angles) % 180
    bin_width = 180 / bins
    quantized_angles = np.floor(angles / bin_width).astype(int)

    return quantized_angles


def compute_histogram(magnitude, angles, cell_size=(8, 8), bins=9):
    # Compute HOG histogram for a cell
    cell_magnitude = magnitude[:cell_size[0], :cell_size[1]]
    cell_angles = angles[:cell_size[0], :cell_size[1]]
    quantized_angles = quantize_orientation(cell_angles, bins)

    histogram = np.zeros(bins)
    for i in range(cell_size[0]):
        for j in range(cell_size[1]):
            histogram[quantized_angles[i, j]] += cell_magnitude[i, j]

    return histogram


def compute_hog_descriptor(image, cell_size=(8, 8), block_size=(2, 2), bins=9):
    magnitude, angles = compute_gradient(image)

    height, width = image.shape
    cell_height, cell_width = cell_size
    block_height, block_width = block_size

    # Calculate the number of cells and blocks
    num_cells_x = width // cell_width
    num_cells_y = height // cell_height
    num_blocks_x = num_cells_x - block_width + 1
    num_blocks_y = num_cells_y - block_height + 1

    hog_descriptor = []

    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            cell_histograms = []
            for y in range(i, i + block_height):
                for x in range(j, j + block_width):
                    cell_histograms.append(compute_histogram(
                        magnitude[y * cell_height:(y + 1) * cell_height, x * cell_width:(x + 1) * cell_width],
                        angles[y * cell_height:(y + 1) * cell_height, x * cell_width:(x + 1) * cell_width],
                        cell_size=cell_size,
                        bins=bins
                    ))
            # Concatenate cell histograms and normalize
            block_histogram = np.concatenate(cell_histograms).ravel()
            block_histogram /= np.linalg.norm(block_histogram)
            hog_descriptor.extend(block_histogram)

    return np.array(hog_descriptor)


if __name__ == "__main__":
    # Load an image
    image = cv.imread('../pic/9.jpg', cv.IMREAD_GRAYSCALE)

    # Compute HOG descriptor
    hog_descriptor = compute_hog_descriptor(image)

    # Print the HOG descriptor
    print(*hog_descriptor)
    # cv.imshow("hog_descriptor", hog_descriptor.resize(image.shape()))
    # cv.waitKey(0)