# Model: Weighted Approximate-Rank Pairwase (WARP)

from lightfm import LightFM

from .base import PredictionModel

from scipy.sparse import coo_matrix
import numpy as np

class WARP(PredictionModel):
    def __init__(self):
        """ Model inicialization 
        """
        self.model = LightFM(loss='warp')
        self.trainset = None
        
    def fit(self,X,y):
        #Create Coo-Matrix with X and y
        data = coo_matrix((y, (X[:,0], X[:,1])))
        self.trainset = np.column_stack((X,y)) #create array [[user_id][item_id][rating]]
        #Fit the model
        self.model.fit(data)

    def predict(self,X):
        test_rating_result = np.zeros(X.shape[0])
        pos = 0
        data_list = list(zip(X[:,0],X[:,1]))
        for (uid,iid) in data_list:
            iid_array = [iid]
            test_rating_result[pos] = np.asscalar(self.model.predict(uid,iid_array))
            pos = pos + 1
        return test_rating_result

    def recommend(self,user_id,N=1):
        #Create array with [user_id] and all [item_id]
        array = np.column_stack((np.repeat(user_id, np.unique(self.trainset[:,1]).shape[0]),np.unique(self.trainset[:,1])))
        #Predict rating
        result = self.predict(array)
        #Create array with [user_id][item_id] and predicted [rating]
        array_result = np.column_stack((array,np.asarray(result)))
        #Order descending (biggest 'rating' first) and them return 'N' first 'item_id' values
        return array_result[array_result[:,2].argsort()[::-1]][:N][:,1]
