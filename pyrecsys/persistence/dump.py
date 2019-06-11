#Persistence: Dump: save a model instance

import pickle

def dump(model,filename):
    """ 
    Write a pickled representation of 'obj' to the open file object 'file'
        
    Parameters : 
        model : model to save
        filename : directory to save the model
    """
    pickle.dump(model,open(filename,'wb'))