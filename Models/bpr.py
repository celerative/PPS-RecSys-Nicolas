# Model: Bayesian Personalized Ranking (BPR)

from lightfm import LightFM
from scipy.sparse import coo_matrix

from .base import Base

class LightFM_BPR(Base):
    def __init__(self):
        """ Model inicialization 
        """
        self.model = LightFM(loss='bpr')

    def fit(self,X,y):
        #Create Coo-Matrix with X and y
        data = coo_matrix((y, (X[:,0], X[:,1])))

        #Fit the model
        self.model.fit(data)

    def predict(self,uid,iid):
        iid_array = [iid]
        return self.model.predict(uid,iid_array)
