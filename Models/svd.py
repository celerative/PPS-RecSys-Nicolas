# Model: Singular Value Decomposition (SVD)

from surprise import SVD as surprise_svd

from .surprise_wrapper import make_surprise_wrapper

SVD = make_surprise_wrapper(surprise_svd)
