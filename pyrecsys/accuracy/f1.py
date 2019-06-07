#Metric: F1

from sklearn.metrics import f1_score

def f1(y_true,y_pred,**kwargs):
    """

    F1: Weighted average of the precision and recall.
    The best value is 1 and the worst value is 0.

    Parameters
        y_true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) target values.

        y_pred : 1d array-like, or label indicator array / sparse matrix
            Estimated targets as returned by a classifier.

    Returns
        f1_score : float or array of float, shape = [n_unique_labels]
            F1 score of the positive class in binary classification or weighted average of the F1 scores of each class for the multiclass task.
    
    """
    #p = precision.precision()
    #r = recall.recall()
    #f1 = 2 * (p * r) / (p + r)
    return f1_score(y_true,y_pred,**kwargs)
