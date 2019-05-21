# Model: K Nearest Neighbors Basic (KNN Basic)

from surprise import KNNBasic

from .surprise_wrapper import SurpriseWrapper

SVD = SurpriseWrapper(KNNBasic)
