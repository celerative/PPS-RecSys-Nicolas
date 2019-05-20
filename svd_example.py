from models.svd import SVD
#import acurracy.rmse as rmse
import pandas as pd

#import sys
#sys.path.insert(0,'C:\\Users\\Nicolas\\Desktop\\Recommender_Library\\modules')

#from svd import SVD

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
model = SVD()

#Train model
model.fit(X,y)

# get a prediction for specific users and items.
uid = 32
iid = 1
pred = model.predict(uid, iid)
print(pred)

#Get RMSE
#accuracy_rmse = rmse(model)
#print("The RMSE is: %d",accuracy_rmse)