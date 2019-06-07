#Metric: precision

from sklearn.metrics import precision_score

def precision(y_true, y_pred,average='binary',**kwargs):
    """ 

    Precision: Ability of the classifier not to label as positive a sample that is negative.
    The best value is 1 and the worst value is 0.
    
    Parameters
        y_true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) target values.

        y_pred : 1d array-like, or label indicator array / sparse matrix
            Estimated targets as returned by a classifier.
        
    Returns	
        precision : float (if average is not None) or array of float, shape = [n_unique_labels]
            Precision of the positive class in binary classification or weighted average of the precision of each class for the multiclass task.

    """
    return precision_score(y_true,y_pred,average,**kwargs)
