import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure


def get_normalized_cdf(hist):
    cdf = hist.cumsum()
    normalized_cdf = cdf / float(cdf.max())
    return normalized_cdf


def get_lookup_table(src_cdf, ref_cdf):
    lookup_table = np.zeros(256)
    lookup_val = 0
    for s_idx, s_val in enumerate(src_cdf):
        for r_idx, r_val in enumerate(ref_cdf):
            if r_val >= s_val:
                lookup_val = r_idx
                break
        lookup_table[s_idx] = lookup_val

    return lookup_table.astype(np.uint8)


def match_hist(src_img, ref_img):
    s_b, s_g, s_r = cv.split(src_img)
    r_b, r_g, r_r = cv.split(ref_img)

    s_hist_b = cv.calcHist([src_img], [0], None, [256], [0, 256])
    s_hist_g = cv.calcHist([src_img], [1], None, [256], [0, 256])
    s_hist_r = cv.calcHist([src_img], [2], None, [256], [0, 256])
    r_hist_b = cv.calcHist([ref_img], [0], None, [256], [0, 256])
    r_hist_g = cv.calcHist([ref_img], [1], None, [256], [0, 256])
    r_hist_r = cv.calcHist([ref_img], [2], None, [256], [0, 256])

    s_cdf_b = get_normalized_cdf(s_hist_b)
    s_cdf_g = get_normalized_cdf(s_hist_g)
    s_cdf_r = get_normalized_cdf(s_hist_r)
    r_cdf_b = get_normalized_cdf(r_hist_b)
    r_cdf_g = get_normalized_cdf(r_hist_g)
    r_cdf_r = get_normalized_cdf(r_hist_r)

    blue_LUT = get_lookup_table(s_cdf_b, r_cdf_b)
    green_LUT = get_lookup_table(s_cdf_g, r_cdf_g)
    red_LUT = get_lookup_table(s_cdf_r, r_cdf_r)

    new_b = cv.LUT(s_b, blue_LUT)
    new_g = cv.LUT(s_g, green_LUT)
    new_r = cv.LUT(s_r, red_LUT)

    new_img = cv.merge([new_b, new_g, new_r])

    return new_img


def plot_histogram(img, title, mask=None):
    # split the image into blue, green and red channels
    channels = cv.split(img)
    colors = ("b", "g", "r")
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    # loop over the image channels
    for (channel, color) in zip(channels, colors):
        # compute the histogram for the current channel and plot it
        hist = cv.calcHist([channel], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])


def main():
    src = cv.imread("Resources\\image.jpg")
    ref = cv.imread("Resources\\image2.jpg")

    new_img = match_hist(src, ref)

    multi = True if src.shape[-1] > 1 else False
    scikit = exposure.match_histograms(src, ref, multichannel=multi)

    cv.imshow("Source", src)
    cv.imshow("Ref", ref)
    cv.imshow("New", new_img)
    cv.imshow("Scikit", scikit)

    # plot_histogram(src, "OG Hist")
    # plot_histogram(ref, "Reference Hist")
    # plot_histogram(new_img, "New Hist")
    # plot_histogram(scikit, "Scikit")

    plt.show()
    cv.waitKey(0)


if __name__ == '__main__':
    main()