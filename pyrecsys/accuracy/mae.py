#Accuracy: Mean Absolute Error (MAE)

from sklearn.metrics import mean_absolute_error

def mae(y_true, y_pred,**kwargs):
    """
    The best value is 0.0.

    Parameters : 
        y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        
        y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)

    Returns	: 
        mae : [float] or [ndarray] of floats
    
    """
    return mean_absolute_error(y_true,y_pred,**kwargs)
