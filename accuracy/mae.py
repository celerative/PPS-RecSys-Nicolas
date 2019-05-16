#Metric: Mean Absolute Error (MAE)

from sklearn.metrics import mean_absolute_error

def mae(y_true, y_pred,multioutput='uniform_average',**kwargs):
    """
    
    The best value is 0.0.

    Parameters
        y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        multioutput : string in [‘raw_values’, ‘uniform_average’]
            or array-like of shape (n_outputs) Defines aggregating of multiple output values. Array-like value defines weights used to average errors.
            ‘raw_values’ : Returns a full set of errors in case of multioutput input.
            ‘uniform_average’ : Errors of all outputs are averaged with uniform weight.

    Returns	
        loss : float or ndarray of floats
    
    """
    return mean_absolute_error(y_true,y_pred,multioutput,**kwargs)
