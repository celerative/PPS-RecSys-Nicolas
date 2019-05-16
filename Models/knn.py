# Model: K Nearest Neighbor (KNN)

from sklearn.neighbors import KNeighborsClassifier

from .base import Base

class KNN(Base):
    def __init__(self,n_neighbors):
        """ Model inicialization 
        """
        self.model = KNeighborsClassifier(n_neighbors)
    
    def fit(self,trainset,y_train=None):
        self.model.fit(trainset,y_train)

    def predict(self,X):
        return self.model.predict(X)
