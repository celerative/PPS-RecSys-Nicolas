# Model: K Nearest Neighbors (KNN)

from surprise import KNNBasic as surprise_knn_basic
from surprise import KNNWithMeans as surprise_knn_with_means
from surprise import KNNBaseline as surprise_knn_baseline
from surprise import KNNWithZScore as surprise_knn_with_z_score

from .surprise_wrapper import SurpriseWrapper

KNNBasic = SurpriseWrapper(surprise_knn_basic) # Model: K Nearest Neighbors Basic (KNN Basic)

KNNWithMeans = SurpriseWrapper(surprise_knn_with_means) # Model: K Nearest Neighbors With Means (KNN With Means)

KNNBaseline = SurpriseWrapper(surprise_knn_baseline) # Model: K Nearest Neighbors Baseline (KNN Baseline)

KNNWithZScore = SurpriseWrapper(surprise_knn_with_z_score) # Model: K Nearest Neighbors With Zero Score (KNN With Z Score)
