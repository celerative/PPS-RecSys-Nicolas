#Metric: Area Under the Curve (AUC)

def auc(y_true, y_score, average='macro'):
    """
    AUC: Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

    Parameters
        y_true : array, shape = [n_samples] or [n_samples, n_classes]
            True binary labels or binary label indicators.

        y_score : array, shape = [n_samples] or [n_samples, n_classes]
            Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions (as returned by “decision_function” on some classifiers). For binary y_true, y_score is supposed to be the score of the class with greater label.
        
        average : string, [None, ‘micro’, ‘macro’ (default), ‘samples’, ‘weighted’]
            If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data:
                'micro': Calculate metrics globally by considering each element of the label indicator matrix as a label.
                'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
                'weighted': Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label).
                'samples': Calculate metrics for each instance, and find their average.
            Will be ignored when y_true is binary.

    Returns	
        auc : float
    
    """
    pass
