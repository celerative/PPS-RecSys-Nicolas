# Model: Singular Value Decomposition (SVD)

from surprise import SVD as surprise_svd
from surprise import dataset
from surprise import Reader
import pandas as pd

from .base import Base

class MyDataset(dataset.DatasetAutoFolds):

    def __init__(self, df, reader):

        self.raw_ratings = [(uid, iid, r, None) for (uid, iid, r) in
                            zip(df.index.values[0], df.columns.values[0], df['rating'])]
        self.reader=reader

class SVD(Base):
    def __init__(self):
        """ Model inicialization 
        """
        self.model = surprise_svd()

    def fit(self,trainset):
        #Convert trainset to Surprise trainset using Pandas
        df = pd.DataFrame.from_dict(trainset)
        reader = Reader(line_format='user item rating', rating_scale=(1, 5))
        data = MyDataset(df, reader)
        trainset_for_surprise = data.build_full_trainset()

        self.model.fit(trainset_for_surprise)

    def predict(self,uid,iid,r_ui=None,clip=True):
        uid_str = str(uid)
        iid_str = str(iid)
        return self.model.predict(uid_str,iid_str,r_ui,clip)
