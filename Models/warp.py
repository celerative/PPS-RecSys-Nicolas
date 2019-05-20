# Model: Weighted Approximate-Rank Pairwase (WARP)

from lightfm import LightFM

from .base import Base
from scipy.sparse import coo_matrix

class WARP(Base):
    def __init__(self):
        """ Model inicialization 
        """
        self.model = LightFM(loss='warp')
        
    def fit(self,X,y):
        #Create Coo-Matrix with X and y
        data = coo_matrix((y, (X[:,0], X[:,1])))

        #Fit the model
        self.model.fit(data)

    def predict(self,uid,iid):
        iid_array = [iid]
        return self.model.predict(uid,iid_array)
