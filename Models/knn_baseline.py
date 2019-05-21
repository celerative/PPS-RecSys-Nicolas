# Model: K Nearest Neighbors Baseline (KNN Baseline)

from surprise import KNNBaseline

from .surprise_wrapper import SurpriseWrapper

SVD = SurpriseWrapper(KNNBaseline)