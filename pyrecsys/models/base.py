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
    def recommend(self,user_id,N=1):
        """
        Recommend any item to the user_id

        Parameters :
            user_id : userID

        Returns :
            item_id : array of N [item_id]
        """
        pass
    
    @abstractmethod
    def get_params(self,deep=True):
        """
        Get parameters for this estimator

        Parameters : 
            deep : [boolean] - If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns : 
            params : [dict] - Parameter names mapped to their values.
        """

class PredictionModel(Model, metaclass=ABCMeta):

    @abstractmethod
    def predict(self,X):
        """
        Predict rating that the user_id would give to the item_id 
        
        Parameters : 
            X : [array] ; [user_id] x [item_id]
            
        Returns : 
            rating : array of N [rating]
        """
        pass