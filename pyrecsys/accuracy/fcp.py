#Accuracy: Fraction of Concordant Pairs (FCP)

from surprise.accuracy import fcp
import numpy as np

def fcp(y_true,y_pred):
    """

    Parameters : 
        y_true : [narray] of [user_id][item_id][rating]

        y_pred : [narray] of [user_id][item_id][rating]
    """
    predictions_u = defaultdict(list)
    nc_u = defaultdict(int)
    nd_u = defaultdict(int)

    for u0, i0, r0 in y_true:
        # Si existe el 'user_id' u0 y el 'item_id' i0 en y_pred
            #est = rating de [u0][i0]
        # Si no
            #est = 0
        predictions_u[u0].append((r0, est))

    for u0, preds in iteritems(predictions_u):
        for r0i, esti in preds:
            for r0j, estj in preds:
                if esti > estj and r0i > r0j:
                    nc_u[u0] += 1
                if esti >= estj and r0i < r0j:
                    nd_u[u0] += 1

    nc = np.mean(list(nc_u.values())) if nc_u else 0
    nd = np.mean(list(nd_u.values())) if nd_u else 0

    return nc / (nc + nd)
