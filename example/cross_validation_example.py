from pyrecsys.models.svd import SVD
from pyrecsys.model_selection.cross_validation import cross_validation

import pandas as pd

#Read File
file_path = 'ratings_small.csv'
df = pd.read_csv(file_path, dtype='unicode')

#Change type of data
df.userId = df.userId.astype(int)
df.movieId = df.movieId.astype(int)
df.rating = df.rating.astype(float)

#Sort Values
df.sort_values(by=['userId','movieId'],ascending=True)

X = df[['userId','movieId']].values
y = df.rating.values

#Create model instance
model = SVD()

score = 'neg_mean_squared_error'
cv = 5
cross = cross_validation(model,X,y,cv=cv,scoring=score)

print("The Cross Validation with score ",score," and cv="+str(cv)+" is: ",cross)