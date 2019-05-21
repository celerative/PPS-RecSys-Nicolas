# Model: K Nearest Neighbors With Zero Score (KNN With Z Score)

from surprise import KNNWithZScore

from .surprise_wrapper import SurpriseWrapper

SVD = SurpriseWrapper(KNNWithZScore)