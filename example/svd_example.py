from pyrecsys.models.svd import SVD
#from pyrecsys.model_selection.cross_validation import cross_validation
from pyrecsys.accuracy.fcp import fcp

import pandas as pd
import numpy as np

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

#Train model
model.fit(X,y)

# get a prediction for specific users and items.
uid = 1
X_uid = X[ X[:,0] == uid ]
pred = model.predict(X_uid)

# get N recommendations for user
#N = 10
#result = model.recommend(uid,N)
#print("the movie_id recommend for the user_id " + str(uid) + " are ", result)

#print(cross_validation(model,X,y,cv=5,scoring='neg_mean_squared_error'))

y_true = np.column_stack((X[:,0],y))
y_true = y_true[y_true[:,0] == uid]
print("FCP: ",fcp(y_true[:,1], pred, X[X[:,0] == uid]))
