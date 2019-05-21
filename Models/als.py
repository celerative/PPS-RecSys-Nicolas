# Model: Alternating Least Square (ALS)

from .base import Model

from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix

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

    def recommend(self,user_id):
        print(self.model.recommend(user_id,self.trainset.tocsr(),N=10))
