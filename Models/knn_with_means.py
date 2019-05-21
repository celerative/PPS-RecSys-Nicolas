# Model: K Nearest Neighbors With Means (KNN With Means)

from surprise import KNNWithMeans

from .surprise_wrapper import SurpriseWrapper

SVD = SurpriseWrapper(KNNWithMeans)