import numpy as np

def segmentation_boundary_length(S):
    '''
    :param S:   Segmentation image
    '''
    lx = S[1:,:] != S[:-1,:]
    ly = S[:,1:] != S[:, :-1]

    return np.sum(lx) + np.sum(ly)

def segmentation_histogram(ax, I, S, edges=None):
    '''
    Histogram for data and each segmentation label.

    :param ax:      Plotting axis
    :param I:       Image
    :param S:       Grid segments to plot
    :param edges:   Graph edges to plot
    '''
    if edges is None:
        edges = np.linspace(I.min(), I.max(), 100)
    ax.hist(I.ravel(), bins=edges, color = 'k')
    centers = 0.5*(edges[:-1] + edges[1:]);
    for k in range(S.max()+1):
        ax.plot(centers, np.histogram(I[S==k].ravel(), edges)[0])

def segmentation_energy(S, I, mu, beta):
    '''
    Computes the 1- and 2-clique potentials

    :param S:       Segmentation configuration
    :param I:       Image to segment
    :param mu:      Intensities for segmentation classes
    :param beta:    Smoothnes term

    :returns:      1- and 2-clique potentials (V1 and V2) 
    '''
    
    # likelihood energy
    V1 = ((mu[S]-I)**2).sum()
    
    # prior energy
    V2x = S[1:,:]!=S[:-1,:]
    V2y = S[:,1:]!=S[:,:-1]
    V2 = beta * (V2x.sum() + V2y.sum())
    
    return(V1,V2)
