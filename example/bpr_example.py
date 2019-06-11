from pyrecsys.models.bpr import LightFM_BPR

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
model = LightFM_BPR()

#Train model
model.fit(X,y)

# get a prediction for specific user.
uid = 1
X_uid = X[ X[:,0] == uid ]
pred = model.predict(X_uid)
print("predictions for the user_id "+str(uid)+" are: ",pred)

# get top N recommendations for the user with [user_id] 'uid'
#N = 10
#recommend = model.recommend(uid,N)
#print("The Top "+str(N)+" recommendations for the user_id "+str(uid)+" are ",recommend)
