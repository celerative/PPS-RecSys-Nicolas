# Base class Surprise for all Metods of Surprise

from surprise import SVD as surprise_svd
from surprise import dataset
from surprise import Reader
import pandas as pd
import numpy as np

from .base import PredictionModel

class MyDataset(dataset.DatasetAutoFolds):

    def __init__(self, df, reader):

        self.raw_ratings = [(uid, iid, r, None) for (uid, iid, r) in
                            zip(df[df.columns.values[0]], df[df.columns.values[1]], df[df.columns.values[2]])]
        self.reader=reader

def make_surprise_wrapper(model):
    class SurpriseWrapper(PredictionModel):
        def __init__(self):
            self.model = model()
            self.trainset = None

        def fit(self,X,y):
            data = {'userId': X[:,0], 'itemId': X[:,1], 'rating': y.values}
            #Convert data to Surprise trainset using Pandas
            df = pd.DataFrame.from_dict(data)
            reader = Reader(line_format='user item rating', rating_scale=(1, 5))
            data = MyDataset(df, reader)
            trainset_for_surprise = data.build_full_trainset()
            self.trainset = trainset_for_surprise #save trainset
            self.model.fit(trainset_for_surprise) #fit model
        
        def predict(self,X):
            rating = [0] * X.shape[0]
            data_list = list(zip(X[:,0],X[:,1],rating))
            test_rating_result = rating
            pos = 0
            for (uid,iid,r_ui_trans) in data_list: 
                test_rating_result[pos] = self.model.predict(uid,iid,r_ui_trans - self.trainset.offset,verbose=False).est
                pos = pos + 1
            return test_rating_result

        def recommend(self,user_id):
            return None

        def get_params(self,deep=True):
            return dict()

    return SurpriseWrapper