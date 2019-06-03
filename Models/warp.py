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
        
    def fit(self,X,y):
        #Create Coo-Matrix with X and y
        data = coo_matrix((y, (X[:,0], X[:,1])))

        #Fit the model
        self.model.fit(data)

    def predict(self,X):
        rating = [0] * X.shape[0]
        data_list = list(zip(X[:,0],X[:,1]))
        test_rating_result = rating
        pos = 0
        for (uid,iid) in data_list:
            iid_array = [iid]
            test_rating_result[pos] = np.asscalar(self.model.predict(uid,iid_array))
            pos = pos + 1
        return test_rating_result

    def recommend(self,user_id):
        return None