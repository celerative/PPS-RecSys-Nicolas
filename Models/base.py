# Base class for all Metods

class Base:

    def fit(self,trainset):
        """ Train an algorithm on a given training set.

            This method is called by every derived class as the first basic step for training an algorithm. 
            It basically just initializes some internal structures and set the self.trainset attribute. 
            
            Parameters
                trainset
            
            Return
                self 
        """
        pass

    def predict(self,X):
        """ Predict using the linear model
        
            Parameters
                X : array_like or sparse matrix, shape (n_samples, n_features)
            
            Returns
                C : array, shape (n_samples,)
        """
        pass

    def load(file, *, fix_imports=True, encoding="ASCII", errors="strict"):
        """ Read a pickled object representation from the open file object 'file' and return the reconstituted object hierarchy specified therein. 
        """
        pass

    def dump(self):
        """ Write a pickled representation of 'self' to the open file object given in the constructor.
            
            Parameters
                self : model to save
        """
        pass