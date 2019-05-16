#from models.svd import SVD
#import acurracy.rmse as rmse
#import pandas as pd

#import sys
#sys.path.insert(0,'C:\\Users\\Nicolas\\Desktop\\Recommender_Library\\modules')

#from svd import SVD

from surprise import Dataset

# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k')

# Retrieve the trainset.
trainset = data.build_full_trainset()

#Load de dataset (example: using pandas)
#data = pd.read_scv('movies_metadata.csv', dtype='unicode')

#Create model instance
model = SVD()

#Train model
model.fit(data)

#Get recommendations
#predictions = model.predict()
#for p in predictions:
#    print("%s",p)

#Get RMSE
#accuracy_rmse = rmse(model)
#print("The RMSE is: %d",accuracy_rmse)