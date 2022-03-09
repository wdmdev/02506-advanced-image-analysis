import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_transformation(P, s, R, t):
    '''
    Transforms the point set P of shape (2 x N).

    :param P:   The point set to transform
    :param s:   The transformation scale
    :param R:   The rotation
    :param t:   The translation

    :returns:   The transformed point set.
    '''

    return s*R @ P + t

def get_transformation(P, Q):
    '''
    Finds scale, rotation, and translation for the 
    registration from point set P to Q.

    :param P:   Point set from
    :param Q:   Points set to

    :returns:   Triple with scale, rotation, and translation
    '''
    mu_p = np.mean(P, axis=1, keepdims=True)
    mu_q = np.mean(Q, axis=1, keepdims=True)

    # Scale
    s = np.sum(np.linalg.norm(Q - mu_q)) / np.sum(np.linalg.norm(P-mu_p))

    # Covariance matrix
    C = (Q - mu_q) @ (P - mu_p).T
    # Singular value decomposition
    U, S, VT = np.linalg.svd(C)
    # Using R_hat and D to avoid reflection when det(R_hat) = -1
    R_hat = U@VT
    D = np.array([[1, 0], [0, np.linalg.det(R_hat)]])
    # Rotation matrix
    R = R_hat@D

    # Translation
    t = mu_q - s*R@mu_p

    return s, R, t

def match_SIFT(im1, im2, thres = 0.6):
    '''
    Extract the coordinates of matching points in the given images

    :param im1:     Match image 1
    :param im2:     Match image 2
    :param thres:   Threshold, points with Euclidean dist greater than this
                    will be discarded

    :returns:       Coordinates of matching points in the given images
    '''
    
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1,None)
    kp2, des2 = sift.detectAndCompute(im2,None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    
    # Apply ratio test
    good_matches = []
    
    for m,n in matches:
        if m.distance/(n.distance+10e-10) < thres:
            good_matches.append([m])
    
    # Find coordinates
    pts_im1 = [kp1[m[0].queryIdx].pt for m in good_matches]
    pts_im1 = np.array(pts_im1, dtype=np.float32).T
    pts_im2 = [kp2[m[0].trainIdx].pt for m in good_matches]
    pts_im2 = np.array(pts_im2, dtype=np.float32).T
    return pts_im1, pts_im2

def plot_matching_keypoints(im1, im2, pts_im1, pts_im2):
    '''
    Plots the matching keypoints in the given images

    :param im1:     Matching image 1
    :param im2:     Matching image 2
    :param pts_im1: Keypoints image 1
    :param pts_im2: Keypoints image 2
    '''
    r1,c1 = im1.shape
    r2,c2 = im2.shape
    n_row = np.maximum(r1, r2)
    n_col = c1 + c2
    im_comp = np.zeros((n_row,n_col))
    im_comp[:r1,:c1] = im1
    im_comp[:r2,c1:(c1+c2)] = im2
    
    fig,ax = plt.subplots(1)
    ax.imshow(im_comp, cmap='gray')
    ax.plot(pts_im1[0],pts_im1[1],'.r')
    ax.plot(pts_im2[0]+c1,pts_im2[1],'.b')
    ax.plot(np.c_[pts_im1[0],pts_im2[0]+c1].T,np.c_[pts_im1[1],pts_im2[1]].T,'w',linewidth = 0.5)