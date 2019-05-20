# Base class for all Metods

from abc import ABCMeta, abstractmethod, abstractclassmethod

class Base(metaclass=ABCMeta):

    @abstractmethod
    def fit(self,X,y):
        """ Train an algorithm on a given training set.

            This method is called by every derived class as the first basic step for training an algorithm. 
            It basically just initializes some internal structures and set the self.trainset attribute. 
            
            Parameters
                X : Array of tuplas ['userId','itemId']
                y : array of ratings for each tupla of X
            
            Return
                self 
        """
        pass

    @abstractmethod
    def predict(self,X):
        """ Predict using the linear model
        
            Parameters
                X : array_like or sparse matrix, shape (n_samples, n_features)
            
            Returns
                C : array, shape (n_samples,)
        """
        pass