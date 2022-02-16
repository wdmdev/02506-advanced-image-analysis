import numpy as np

def segmentation_boundary_length(S):
    '''
    :param S:   Segmentation image
    '''
    lx = S[1:,:] != S[:-1,:]
    ly = S[:,1:] != S[:, :-1]

    return np.sum(lx) + np.sum(ly)
