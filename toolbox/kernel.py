import numpy as np
from scipy.ndimage import convolve

def gaussian_1d(x, sigma):
    '''
    Analytical 1D Gaussian kernel
    '''
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2))


def dgaussian_1d(x, sigma):
    '''
    Analytical 1st order derivative of Gaussian kernel
    '''
    return -x/(sigma**3 * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2))

def gaussian_2d(x, y, sigma):
    '''
    Analytical 2D Gaussian kernel
    '''
    variance = sigma**2
    return 1/(2*variance*np.pi) * np.exp(-(x**2 + y**2)/(2*variance))

def convolve_img(img, kernel):
    tmp_img = img.copy()

    if len(img.shape) == 2:
        return convolve(img, kernel)
    else:
        for layer in range(img.shape[2]):
            tmp_img[:, :, layer] = convolve(img[:,:,layer], kernel)
    
    return tmp_img