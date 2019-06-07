#Method: Cosine()

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

def cosine(X,Y=None,dense_output=True):
    """
    
    Compute the cosine similarity between all pairs of users (or items)

    Parameters:
        X : ndarray or sparse array, shape: (n_samples_X, n_features)
            Input data.

        Y : ndarray or sparse array, shape: (n_samples_Y, n_features)
            Input data. If None, the output will be the pairwise similarities between all samples in X.

        dense_output : boolean (optional), default True
            Whether to return dense output even when the input is sparse. If False, the output is sparse if both input arrays are sparse.

    Return:
        kernel matrix : array
            An array with shape (n_samples_X, n_samples_Y).

    """
    #Reemplace NaN with Zero
    X[np.isnan(X)] = 0
    if Y is not None:
        Y[np.isnan(Y)] = 0
        
    return cosine_similarity(X,Y,dense_output)