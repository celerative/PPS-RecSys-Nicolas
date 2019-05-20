# Model: Alternating Least Square (ALS)

from .base import Base

from implicit.als import alternating_least_squares
from scipy.sparse import coo_matrix

class ALS(Base):
    def __init__(self):
        """ Model inicialization 
        """
        self.model = alternating_least_squares()

    def fit(self,X,y):
        #Create Coo-Matrix with X and y
        data = coo_matrix((y, (X[:,0], X[:,1])))

        data.transpose() #rows:[n_items] ; columns:[n_users]
        
        self.model.fit(data)

    def predict(self,X):
        pass
