from pyrecsys.metrics.pearson import pearson

import numpy as np
from numpy import nan

def test_pearson():
    #Create two arrays of user_rating
    X = np.array([[nan,5,1]]) # Ratings of userId 1
    Y = np.array([[1,nan,1]]) # Ratings of userId 2

    #Calcule the Pearson Similarity
    result = pearson(X,Y)
    assert result >= -1
    assert result <= 1