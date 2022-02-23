import cv2
import numpy as np
import skimage.feature
from toolbox.kernel import gaussian_1d, ddgaussian_1d, convolve_img

def laplacian_image(img, sigma=3, kernel_size=5):
    '''
    Scale-space Laplacian.

    :param img:     Image for which to compute the Laplacian

    :returns:       Scale-space Laplacian of image
    '''
    g = gaussian_1d(sigma, kernel_size)
    ddg = ddgaussian_1d(sigma, kernel_size)
    Lxx = cv2.filter2D(cv2.filter2D(img, -1, g), -1, ddg.T)
    Lyy = cv2.filter2D(cv2.filter2D(img, -1, ddg), -1, g.T)

    return (sigma**2) * (Lxx + Lyy)

def scale_space_laplacian_detection(L_img, threshold):
    '''
    Scale invariant blob detection using the scale-space Laplacian.

    :param img:     Image in which to detect the blobs

    :returns:       Image coordinates of detected blobs
    '''
    coord_pos = skimage.feature.peak_local_max(L_img, threshold_abs=threshold)
    coord_neg = skimage.feature.peak_local_max(-L_img, threshold_abs=threshold)
    coords = np.r_[coord_pos, coord_neg]
    return coords
