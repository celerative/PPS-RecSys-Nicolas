# Model: Bayesian Personalized Ranking (BPR)

from lightfm import LightFM
from scipy.sparse import coo_matrix

from .base import PredictionModel

import numpy as np

class LightFM_BPR(PredictionModel):
    def __init__(self):
        """ Model inicialization 
        """
        self.model = LightFM(loss='bpr')

    def fit(self,X,y, **fit_params):
        print(fit_params)
        #Create Coo-Matrix with X and y
        data = coo_matrix((y, (X[:,0], X[:,1])))
        #print("X: ",X.shape)
        #print("Users: ",X[:,0].shape)
        #print("Items: ",X[:,1].shape)
        X = np.require(X,requirements=['C','O','W','A'])
        y = np.require(y,requirements=['C','W','A'])
        print("X flags: ")
        print(X.flags)
        print("y flags: ")
        print(y.flags)
        #Fit the model
        self.model.fit(data)

    def predict(self,uid,iid):
        iid_array = [iid]
        return self.model.predict(uid,iid_array)

    def recommend(self,user_id):
        return None

    def score(self, X, y=None):
        # counts number of values bigger than mean
        return 1

    def get_params(self,deep=True):
        return dict()