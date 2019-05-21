from models.als import ALS
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
model = ALS()

#Train model
model.fit(X,y)

# get a prediction for specific users and items.
uid = 1
iid = 31
rec = model.recommend(uid)
print("the movie_id recommend for the user_id " + str(uid) + " is " + str(rec))