# Base class for all Metods

from abc import ABCMeta, abstractmethod, abstractclassmethod

class Model(metaclass=ABCMeta):

    @abstractmethod
    def fit(self,X,y):
        """ 
        
        Train an algorithm on a given training set.

            This method is called by every derived class as the first basic step for training an algorithm. 
            It basically just initializes some internal structures and set the self.trainset attribute. 
            
        Parameters:
            X : Array of tuplas ['userId','itemId']
            y : array of ratings for each tupla of X

        """
        pass
    
    @abstractmethod
    def recommend(self,user_id):
        """ Recommend any item to the user_id

        Parameters:
            user_id : userID

        """
        pass

class PredictionModel(Model, metaclass=ABCMeta):

    @abstractmethod
    def predict(self,X):
        """ Predict rating that the user_id would give to the item_id 
        
        Parameters:
            X : [array] ; [user_id] x [item_id]
            
        Returns
            C : list, [rating]
        """
        pass