# Model: K Nearest Neighbor (KNN)

from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing

from .base import Base

class KNN(Base):
    def __init__(self,n_neighbors):
        """ Model inicialization 
        """
        self.model = KNeighborsClassifier(n_neighbors)
    
    def fit(self,X,y=None):
        print(y.values)
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        #y = y.astype('float')
        print(y)
        self.model.fit(X,y.values)

    def predict(self,X):
        return self.model.predict(X)
