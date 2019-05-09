from models.svd import SVD
import acurracy.rmse as rmse
import pandas as pd

#Load de dataset (example: using pandas)
data = pd.read_scv('movies_metadata.csv', dtype='unicode')

#Create model instance
model = SVD()

#Train model
model.fit(data)

#Get recommendations
predictions = model.predict()
for p in predictions:
    print("%s",p)

#Get RMSE
accuracy_rmse = rmse(model)
print("The RMSE is: %d",accuracy_rmse)