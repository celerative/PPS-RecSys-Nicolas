import numpy as np
from numpy import nan

from metrics.cosine import cosine

#Create two arrays of user_rating
X = np.array([[nan,5,1]]) # Ratings of userId 1
print("User rating 1:")
print(X)
Y = np.array([[1,nan,1]]) # Ratings of userId 2
print("User rating 2:")
print(Y)

#Calcule the Cosine Similarity
result = cosine(X,Y)
print("Array result:")
print(result)
