#Metric: Area Under the Curve (AUC)

from sklearn.metrics import roc_auc_score

def auc(y_true, y_score,**kwargs):
    """
    AUC: Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

    Parameters
        y_true : array, shape = [n_samples] or [n_samples, n_classes]
            True binary labels or binary label indicators.

        y_score : array, shape = [n_samples] or [n_samples, n_classes]
            Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions (as returned by “decision_function” on some classifiers). For binary y_true, y_score is supposed to be the score of the class with greater label.

    Returns	
        auc : float
    
    """
    return roc_auc_score(y_true,y_score,**kwargs)
