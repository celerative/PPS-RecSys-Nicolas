# Model: Weighted Approximate-Rank Pairwase (WARP)

from lightfm import LightFM

from .base import Base

class WARP(Base):
    def __init__(self):
        """ Model inicialization 
        """
        self.model = LightFM(loss='warp')
        
    def fit(self,trainset):
        #Convert type object to int and float
        df = trainset
        df.userId = df.userId.astype(int)
        df.movieId = df.movieId.astype(int)
        df.rating = df.rating.astype(float)

        df.sort_values(by=['userId','movieId'],ascending=True)

        table = pd.pivot_table(df,values='rating',index=['userId'],columns=['movieId']) #create table
        table = table.fillna(0) #change NaNs with 0
        table = coo_matrix(table.values) #Create coo_matrix
        table.eliminate_zeros() #eliminate 0
        self.model.fit(table)

    def predict(self,uid,iid):
        iid_array = [iid]
        return self.model.predict(uid,iid_array)
