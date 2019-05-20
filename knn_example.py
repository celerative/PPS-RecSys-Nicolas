import pandas as pd
from sklearn.model_selection import train_test_split

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
model = KNN(n_neighbors = 3)

# Fit the classifier to the data
model.fit(X,y)

# Testing model - Show first 5 model predictions on the test data
array_result = model.predict(X)[0:5]
for a in array_result:
    if(a == 0):
        print("The pacient is not diabetic")
    else:
        print("The pacient is diabetic")