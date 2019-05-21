# Base class Surprise for all Metods of Surprise

from surprise import SVD as surprise_svd
from surprise import dataset
from surprise import Reader
import pandas as pd

from .base import Base

class MyDataset(dataset.DatasetAutoFolds):

    def __init__(self, df, reader):

        self.raw_ratings = [(uid, iid, r, None) for (uid, iid, r) in
                            zip(df.columns.values[0], df.columns.values[1], df['rating'])]
        self.reader=reader

class SurpriseWrapper(Base):
    def __init__(self, model):
        self.model = model()
        self.trainset = None

    def fit(self,X,y):
        data = {'userId': X[:,0], 'itemId': X[:,1], 'rating': y.values}
        #Convert data to Surprise trainset using Pandas
        df = pd.DataFrame.from_dict(data)
        reader = Reader(line_format='user item rating', rating_scale=(1, 5))
        data = MyDataset(df, reader)
        trainset_for_surprise = data.build_full_trainset()
        self.trainset = trainset_for_surprise
        self.model.fit(trainset_for_surprise)
    
    def predict(self,uid,iid):
        inner_uid = self.trainset.to_inner_uid(str(uid))
        inner_iid = self.trainset.to_inner_iid(str(iid))
        return self.model.predict(inner_uid,inner_iid).est   