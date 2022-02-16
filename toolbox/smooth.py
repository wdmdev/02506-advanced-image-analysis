import numpy as np

def simple_curve_smoothning(_lambda, L, X):
    '''
    :_lambda:   Weight controlling the magnitude of the displacement
    :L:         Displacement kernel see week 1 lecture notes page 9
    :X:         Matrix representation of image to smoothen
    '''
    return (np.identity(X.shape) + _lambda*L) @ X
