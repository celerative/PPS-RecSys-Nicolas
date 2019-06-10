import numpy as np
from numpy import nan

from pyrecsys.metrics.cosine import cosine

def test_cosine():
    #Create two arrays of user_rating
    X = np.array([[nan,5,1]]) # Ratings of userId 1
    Y = np.array([[1,nan,1]]) # Ratings of userId 2

    #Calcule the Cosine Similarity
    result = cosine(X,Y)
    assert result >= -1
    assert result <= 1