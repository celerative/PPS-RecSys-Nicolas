from models.global_average import GlobalAverage

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
y = df.rating

#Create model instance
model = GlobalAverage()

#Train model
model.fit(X,y)

#Recommend
uid = 1
N = 10
rec = model.recommend(uid,N)
print("The Top "+str(N)+" recommendations for the user_id "+str(uid)+" are movies with ids: ",rec)

#Predict
pre = model.predict(uid)
print("The rating predicted for the user_id "+str(uid)+" is: "+str(pre))