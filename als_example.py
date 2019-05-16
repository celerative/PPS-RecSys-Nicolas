from models.als import ALS
import pandas as pd

# #Create data
# userID = [9, 32, 2, 45, 1]
# itemID = [1, 1, 1, 2, 2]
# rating = [1, 1, 1, 1, 1]
# data = coo_matrix((rating, (userID, itemID)))
# data.eliminate_zeros()
# print(data.shape)
file_path = 'C:\\Users\\Nicolas\\Desktop\\ratings_small.csv'
data = pd.read_csv(file_path, dtype='unicode')

#Create model instance
model = ALS()

#Train model
model.fit(data)

# get a prediction for specific users and items.
uid = 1
iid = 31
#pred = model.predict(uid, iid)
#print("the prediction for the user " + str(uid) + " and the item " + str(iid) + " is " + str(pred))