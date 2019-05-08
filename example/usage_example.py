from mylib.models import Algorithm
import mylib.acurracy as Acurracy
import pandas as pd

#Load de dataset (example: using pandas)
data = pd.read_scv('movies_metadata.csv', dtype='unicode')

#Create model instance
model = Algorithm()

#Train model
model.Train(data)

#Get recommendations
predictions = model.getRecommendations()
for p in predictions:
    print("%s",p)

#Get RMSE
rmse = Acurracy.rmse(model)
print("The RMSE is: %d",rmse)