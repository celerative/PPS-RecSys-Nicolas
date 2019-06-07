import pandas as pd

from pyrecsys.models.knn import KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore

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

# Create KNN model
model = KNNBasic()
#model = KNNBaseline()
#model = KNNWithMeans()
#model = KNNWithZScore()

# Fit model
model.fit(X,y)

# Testing model 
#result = model.predict(X)
uid = 1
N = 10
result = model.recommend(uid,N)
#print(result)
