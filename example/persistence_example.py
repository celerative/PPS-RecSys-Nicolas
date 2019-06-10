#Example to use persistence functions

from pyrecsys.models.bpr import LightFM_BPR

from pyrecsys.persistence.dump import dump
from pyrecsys.persistence.load import load

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
model = LightFM_BPR()

#Train model
model.fit(X,y)

#Save model
dump(model,'bpr_save.sav')

# get a prediction using original model
pred = model.predict(X)
print("Original Model: the predictions are: " + str(pred))

#Load model
model_load = load('bpr_save.sav')

# get a prediction using the load model
pre = model_load.predict(X)
print("Loaded Model: the predictions are: " + str(pred))