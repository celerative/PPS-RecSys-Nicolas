from models.warp import WARP
from scipy.sparse import coo_matrix

#Create data
userID = [9, 32, 2, 45, 1]
itemID = [1, 1, 1, 2, 2]
rating = [1, 1, 1, 1, 1]
data = coo_matrix((rating, (userID, itemID)))
data.eliminate_zeros()

#Create model instance
model = WARP()

#Train model
model.fit(data)

# get a prediction for specific users and items.
uid = 32
iid = 1
pred = model.predict(uid, iid)
print("the prediction for the user " + str(uid) + " and the item " + str(iid) + " is " + str(pred))