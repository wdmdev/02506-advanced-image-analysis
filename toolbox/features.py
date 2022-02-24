import numpy as np
import scipy.ndimage

def get_gauss_feat_im(im, sigma=1, normalize=True):
    """Gauss derivative feaures for every image pixel.
    Arguments:
        image: a 2D image, shape (r,c).
        sigma: standard deviation for Gaussian derivatives.
        normalize: flag indicating normalization of features.
    Returns:
        imfeat: 3D array of size (r,c,15) with a 15-dimentional feature
             vector for every pixel in the image.
    Author: vand@dtu.dk, 2020
    """
      
    r,c = im.shape
    imfeat = np.zeros((r,c,15))
    imfeat[:,:,0] = scipy.ndimage.gaussian_filter(im,sigma,order=0)
    imfeat[:,:,1] = scipy.ndimage.gaussian_filter(im,sigma,order=[0,1])
    imfeat[:,:,2] = scipy.ndimage.gaussian_filter(im,sigma,order=[1,0])
    imfeat[:,:,3] = scipy.ndimage.gaussian_filter(im,sigma,order=[0,2])
    imfeat[:,:,4] = scipy.ndimage.gaussian_filter(im,sigma,order=[1,1])
    imfeat[:,:,5] = scipy.ndimage.gaussian_filter(im,sigma,order=[2,0])
    imfeat[:,:,6] = scipy.ndimage.gaussian_filter(im,sigma,order=[0,3])
    imfeat[:,:,7] = scipy.ndimage.gaussian_filter(im,sigma,order=[1,2])
    imfeat[:,:,8] = scipy.ndimage.gaussian_filter(im,sigma,order=[2,1])
    imfeat[:,:,9] = scipy.ndimage.gaussian_filter(im,sigma,order=[3,0])
    imfeat[:,:,10] = scipy.ndimage.gaussian_filter(im,sigma,order=[0,4])
    imfeat[:,:,11] = scipy.ndimage.gaussian_filter(im,sigma,order=[1,3])
    imfeat[:,:,12] = scipy.ndimage.gaussian_filter(im,sigma,order=[2,2])
    imfeat[:,:,13] = scipy.ndimage.gaussian_filter(im,sigma,order=[3,1])
    imfeat[:,:,14] = scipy.ndimage.gaussian_filter(im,sigma,order=[4,0])

    if normalize:
        imfeat -= np.mean(imfeat, axis=(0,1))
        imfeat /= np.std(imfeat, axis=(0,1))
    
    return imfeat

def get_gauss_feat_multi(im, sigma = [1,2,4], normalize = True):
    '''Multi-scale Gauss derivative feaures for every image pixel.
    Arguments:
        image: a 2D image, shape (r,c).
        sigma: list of standard deviations for Gaussian derivatives.
        normalize: flag indicating normalization of features.
    Returns:
        imfeat: a a 3D array of size (r*c,n_scale,15) with n_scale features in each pixels, and
             n_scale is length of sigma. Each pixel contains a feature vector and feature
             image is size (r,c,15*n_scale).
    Author: abda@dtu.dk, 2021

    '''
    imfeats = []
    for i in range(0,len(sigma)):
        feat = get_gauss_feat_im(im, sigma[i], normalize)
        imfeats.append(feat.reshape(-1,feat.shape[2]))
    
    imfeats = np.asarray(imfeats).transpose(1,0,2)
    return imfeats