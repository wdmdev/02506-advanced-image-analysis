import numpy as np

def simple_curve_smoothing(_lambda, L, X):
    '''
    :param _lambda:   Weight controlling the magnitude of the displacement
    :param L:         Displacement kernel see week 1 lecture notes page 9
    :param X:         Matrix representation of image to smoothen
    '''
    return (np.eye(X.shape[0]) + _lambda*L) @ X

def iter_simple_curve_smoothing(_lambda, L, X, n_iter):
    '''
    :param _lambda:   Weight controlling the magnitude of the displacement
    :param L:         Displacement kernel see week 1 lecture notes page 9
    :param X:         Matrix representation of image to smoothen
    '''
    return np.linalg.matrix_power(_lambda*L+np.eye(X.shape[0]),n_iter) @ X


def implicit_curve_smoothing(_lambda, L, X):
    '''
    :param _lambda:   Weight controlling the magnitude of the displacement
    :param L:         Displacement kernel see week 1 lecture notes page 9
    :param X:         Matrix representation of image to smoothen
    '''
    return np.linalg.inv(np.eye(X.shape[0]) - _lambda*L) @ X

def elasticity_rigidity_smoothing(alpha, beta, A, B, X):
    '''
    :param alpha:     Elasticity kernel weight
    :param beta:      Rigidity kernel weight
    :param A:         Elasticity kernel
    :param B:         Rigidity kernel
    :param X:         Matrix representation of image to smoothen
    '''
    return np.linalg.inv(np.eye(X.shape[0]) - alpha*A - beta*B) @ X
