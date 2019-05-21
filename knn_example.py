import pandas as pd

from models.knn import KNN

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

# Create KNN model
model = KNN()

# Fit model
model.fit(X,y)

# Testing model 
uid = 32
iid = 1
pred = model.predict(uid,iid)
print("The rating predict for the user " + str(uid) + " and the movie " + str(iid) + " is " + str(pred))