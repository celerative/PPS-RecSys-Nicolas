#Method: Load: load a model instance

import pickle

def load(filename):
    """ 

    Read a string from the open file object 'file' and interpret it as a pickle data stream, reconstructing and returning the original object hierarchy. 
    
    Parameters:
        filename : name of file to read


    """

    return pickle.load(open(filename,'rb'))
