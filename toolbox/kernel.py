import numpy as np
from scipy.ndimage import convolve

def gaussian_1d(sigma, size=5):
    '''
    Analytical 1D Gaussian kernel

    :param sigma:   Standard deviation
    :param size:    Kernel size
    '''
    s = np.ceil(np.max([sigma*size, size]))
    x = np.arange(-s,s+1)
    x = x.reshape(x.shape + (1,))
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2))


def dgaussian_1d(sigma, size=5):
    '''
    Analytical 1st order derivative of Gaussian kernel

    :param sigma:   Standard deviation
    :param size:    Kernel size
    '''
    s = np.ceil(np.max([sigma*size, size]))
    x = np.arange(-s,s+1)
    x = x.reshape(x.shape + (1,))
    return -x/(sigma**3 * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2))

def ddgaussian_1d(sigma,size=5):
    '''
    Analytical 2nd order derivative of Gaussian kernel

    :param sigma:   Standard deviation
    :param size:    Kernel size
    '''
    s = np.ceil(np.max([sigma*size, size]))
    x = np.arange(-s,s+1)
    x = x.reshape(x.shape + (1,))
    return (np.exp(-x**2/(2*sigma**2)*(x-sigma)*(x+sigma)))/(np.sqrt(2*np.pi)*sigma**5)


def gaussian_2d(x, y, sigma):
    '''
    Analytical 2D Gaussian kernel

    :param x:       x-coordinate values
    :param y:       y-coordinate values
    :param sigma:   Standard deviation
    '''
    variance = sigma**2
    return 1/(2*variance*np.pi) * np.exp(-(x**2 + y**2)/(2*variance))

def neighbor_avg_point_displacement(N, diag_val=-2):
    '''
    Creates a kernel that will perform image displacement
    based on the average of the neighboring pixel values.
    Used for curve smoothing on a coordinate input matrix 
    X of shape (N x 2) i.e. N data points for (x,y) coordinates.

    :param N:           Number of coordinate data points i.e. equal to X.shape[0]
    :param diag_val:    Can be used to change the diagonal value of the returned L 
                        matrix. For instance used in the elasticity and rigidity smoothing 
                        for matrix B.
    '''
    row_vec = np.zeros(N)
    row_vec[-1], row_vec[0], row_vec[1] = 1, diag_val, 1

    L = np.zeros((N,N))
    L[0] = row_vec

    for i in range(1, N):
        L[i] = np.roll(row_vec, i)
    
    return L

def convolve_img(img, kernel):
    '''
    Function for convolving image with multiple layers e.g. RGB image

    :param img:     Image to convolve
    :param kernel:  Kernel to convolve image by
    '''
    tmp_img = img.copy()

    if len(img.shape) == 2:
        return convolve(img, kernel)
    else:
        for layer in range(img.shape[2]):
            tmp_img[:, :, layer] = convolve(img[:,:,layer], kernel)
    
    return tmp_img