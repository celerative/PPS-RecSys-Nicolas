#Accuracy: Recall

from sklearn.metrics import recall_score

def recall(y_true,y_pred, **kwargs):
    """ 
    Recall: Ability of the classifier to find all the positive samples.
    The best value is 1 and the worst value is 0.
    
    Parameters : 
        y_true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) target values.

        y_pred : 1d array-like, or label indicator array / sparse matrix
            Estimated targets as returned by a classifier.

    Returns : 
        recall : float (if average is not None) or array of float, shape = [n_unique_labels]
            Recall of the positive class in binary classification or weighted average of the recall of each class for the multiclass task.
    """
    return recall_score(y_true,y_pred, **kwargs)
