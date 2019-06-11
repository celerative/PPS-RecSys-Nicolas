# Model: User Average

from .base import PredictionModel

import pandas as pd
import numpy as np

class UserAverage(PredictionModel):
    def __init__(self):
        """ Model inicialization 
        """
        self.user_average = None
        self.trainset = None
        #self.sort_user_average = None

    def fit(self,X,y):
        #Convert numpy.array to Pandas.Dataframe
        data = {'userId': X[:,0], 'itemId': X[:,1], 'rating': y.values}
        self.trainset = pd.DataFrame.from_dict(data)
        self.user_average = self.trainset.groupby('userId')['rating'].mean() # user average

    def recommend(self,user_id,N=1):        
        return self.trainset['itemId'].unique()[:N]

    def predict(self,user_id):
        return self.user_average[user_id]

    def get_params(self,deep=True):
        return dict()