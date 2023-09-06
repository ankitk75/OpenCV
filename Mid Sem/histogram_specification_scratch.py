import cv2
import numpy as np

def histogram_specification(input_image, reference_image):
    # Read the input and reference images
    input_img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    reference_img = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)

    # Calculate histograms for both images
    input_hist = cv2.calcHist([input_img], [0], None, [256], [0, 256])
    reference_hist = cv2.calcHist([reference_img], [0], None, [256], [0, 256])

    # Normalize histograms
    input_hist /= input_img.size
    reference_hist /= reference_img.size

    # Calculate cumulative distribution functions (CDF)
    input_cdf = input_hist.cumsum()
    reference_cdf = reference_hist.cumsum()

    # Create a lookup table
    lut = np.interp(input_cdf, reference_cdf, range(256))

    # Apply the lookup table to the input image
    output_img = cv2.LUT(input_img, lut)

    return output_img

if __name__ == "__main__":
    input_image_path = "../../Resources/3.jpg" # Replace with your input image path
    reference_image_path = "../../Resources/7.jpg"  # Replace with your reference image path

    output_image = histogram_specification(input_image_path, reference_image_path)

    # Display and save the output image
    cv2.imshow("Output Image", output_image)
    cv2.waitKey(0)
    cv2.imwrite("output_image.jpg", output_image)
    cv2.destroyAllWindows()


