# Model: Alternating Least Square (ALS)

from .base import Model

from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
import numpy as np

class ALS(Model):
    def __init__(self):
        """ Model inicialization 
        """
        self.model = AlternatingLeastSquares()
        self.trainset = None

    def fit(self,X,y):
        #Create Coo-Matrix with X and y
        data = coo_matrix((y, (X[:,0], X[:,1])))
        self.trainset = data
        data.transpose() #rows:[n_items] ; columns:[n_users]
        self.model.fit(data)

    def recommend(self,user_id,N=1):
        n_recomendation = self.model.recommend(user_id,self.trainset.tocsr(),N=N) #array of tuples (item_id,rating)
        #convert array of [tuples] in array of [item_id]
        result = np.zeros(N, dtype=int)
        pos = 0
        for recomendation_tuple in n_recomendation:
            result[pos] = recomendation_tuple[0]
            pos = pos + 1
        return result

    def get_params(self,deep=True):
        return dict()