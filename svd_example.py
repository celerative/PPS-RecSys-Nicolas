from models.svd import SVD
#import acurracy.rmse as rmse
#import pandas as pd

#import sys
#sys.path.insert(0,'C:\\Users\\Nicolas\\Desktop\\Recommender_Library\\modules')

#from svd import SVD

#Load de dataset (example: using pandas)
#data = pd.read_scv('movies_metadata.csv', dtype='unicode')
data = {'userID': [9, 32, 2, 45, 1],
                'itemID': [1, 1, 1, 2, 2],
                'rating': [3, 2, 4, 3, 1]}

#Create model instance
model = SVD()

#Train model
model.fit(data)

# get a prediction for specific users and items.
uid = 32
iid = 1
pred = model.predict(uid, iid)
print(pred)

#Get RMSE
#accuracy_rmse = rmse(model)
#print("The RMSE is: %d",accuracy_rmse)