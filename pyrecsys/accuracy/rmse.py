#Accuracy: Root Mean Square Error (RMSE)

from sklearn.metrics import mean_squared_error
import numpy as np

def rmse(y_true,y_pred,**kwargs):
    """
    RMSE: Is calculated by looking at predicted ratings versus their hidden ground-truth

    Parameters : 
        y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        
        y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)

    Return : 
        rmse : (float) - The Root Mean Squared Error of predictions.
    """
    return np.sqrt(mean_squared_error(y_true,y_pred,**kwargs))
