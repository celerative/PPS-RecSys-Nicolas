#Accuracy: Fraction of Concordant Pairs (FCP)

from surprise.accuracy import fcp
import numpy as np
from collections import defaultdict

def fcp(y_true,y_pred,X=None):
    """

    Parameters : 
        y_true : [narray] of [rating]

        y_pred : [narray] of [rating_estimated]

        X : [narray] (optional) of [user_id][item_id]

    Returns : 
        fcp : [float]
    """
    predictions_u = defaultdict(list)
    nc_u = defaultdict(int)
    nd_u = defaultdict(int)

    y = np.column_stack((y_true,y_pred))
    data = np.column_stack((X[:,0],y))

    for u0, r0, est in data:
        predictions_u[u0].append((r0, est))

    for u0, preds in predictions_u.items():
        for r0i, esti in preds:
            for r0j, estj in preds:
                if esti > estj and r0i > r0j:
                    nc_u[u0] += 1
                if esti >= estj and r0i < r0j:
                    nd_u[u0] += 1

    nc = np.mean(list(nc_u.values())) if nc_u else 0
    nd = np.mean(list(nd_u.values())) if nd_u else 0

    return nc / (nc + nd)
