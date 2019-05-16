# Model: Alternating Least Square (ALS)

from .base import Base

from implicit.als import alternating_least_squares
from scipy.sparse import coo_matrix

class ALS(Base):
    def __init__(self):
        """ Model inicialization 
        """
        self.model = alternating_least_squares()

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
        table.transpose() #rows:[n_items] ; columns:[n_users]
        self.model.fit(table)

    def predict(self,X):
        pass
