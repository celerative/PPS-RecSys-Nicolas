#Method: Pearson()

from pyrecsys.metrics.cosine import cosine

import numpy as np

def pearson(X,Y=None,dense_output=True):
    """
    
    Compute the Pearson correlation coefficient between all pairs of users (or items).

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
    #Calcule user_rating - average
    X = X - X[~np.isnan(X)].mean() # X - X_average
    if Y is not None:
        Y = Y - Y[~np.isnan(Y)].mean() # Y - Y_average
    return cosine(X,Y,dense_output)