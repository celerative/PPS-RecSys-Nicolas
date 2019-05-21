# Model: Singular Value Decomposition (SVD)

from surprise import SVD as surprise_svd

from .surprise_wrapper import SurpriseWrapper

SVD = SurpriseWrapper(surprise_svd)
