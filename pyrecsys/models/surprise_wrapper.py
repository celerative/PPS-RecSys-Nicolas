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
            self.surprise_trainset = None
            self.pandas_trainset = None

        def fit(self,X,y):
            data = {'userId': X[:,0], 'itemId': X[:,1], 'rating': y}
            #Convert data to Surprise trainset using Pandas
            df = pd.DataFrame.from_dict(data)
            reader = Reader(line_format='user item rating', rating_scale=(1, 5))
            data = MyDataset(df, reader)
            trainset_for_surprise = data.build_full_trainset()
            self.pandas_trainset = df
            self.surprise_trainset = trainset_for_surprise #save surprise trainset
            self.model.fit(trainset_for_surprise) #fit model
        
        def predict(self,X):
            rating = [0] * X.shape[0]
            test_rating_result = np.zeros(X.shape[0])
            pos = 0
            data_list = list(zip(X[:,0],X[:,1],rating))
            for (uid,iid,r_ui_trans) in data_list: 
                test_rating_result[pos] = self.model.predict(uid,iid,r_ui_trans - self.surprise_trainset.offset,verbose=False).est
                pos = pos + 1
            return test_rating_result

        def recommend(self,user_id,N=1):
            #Create array with [user_id] and all [item_id]
            array = np.column_stack((np.repeat(user_id, self.pandas_trainset['itemId'].unique().shape[0]),self.pandas_trainset['itemId'].unique()))
            #Predict rating
            result = self.predict(array)
            #Create array with [user_id][item_id] and predicted [rating]
            array_result = np.column_stack((array,np.asarray(result)))
            #Order descending (biggest 'rating' first) and them return 'N' first 'item_id' values
            return array_result[array_result[:,2].argsort()[::-1]][:N][:,1]

        def get_params(self,deep=True):
            return dict()

    return SurpriseWrapper