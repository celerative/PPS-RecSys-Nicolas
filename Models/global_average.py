# Model: Global Average

from .base import PredictionModel

import pandas as pd

class GlobalAverage(PredictionModel):
    def __init__(self):
        """ Model inicialization 
        """
        self.globalAverage = None
        self.trainset = None
    
    def fit(self,X,y):
        self.globalAverage = y.values.sum() / y.values.size # average rating
        
        #Convert numpy.array to Pandas.Dataframe
        data = {'userId': X[:,0], 'itemId': X[:,1], 'rating': y.values}
        self.trainset = pd.DataFrame.from_dict(data)

    def recommend(self,user_id,N=1):
        return self.trainset['itemId'].unique()[:N]

    def predict(self,user_id):
        return self.globalAverage
