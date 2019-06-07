# Model: Item Average

from .base import PredictionModel

import pandas as pd

class ItemAverage(PredictionModel):
    def __init__(self):
        """ Model inicialization 
        """
        self.itemAverage = None
        self.trainset = None

    def fit(self,X,y):
        #Convert numpy.array to Pandas.Dataframe
        data = {'userId': X[:,0], 'itemId': X[:,1], 'rating': y.values}
        self.trainset = pd.DataFrame.from_dict(data)
        self.itemAverage = self.trainset.groupby('itemId')['rating'].mean() # item average

    def recommend(self,user_id,N=1): #recommend N users for de user with id: user_id
        return self.trainset['itemId'].unique()[:N]