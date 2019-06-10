# Recommender System Library
PPS: Practica Profesional Supervisadas
Celerative

## Install
First, you need install [Numpy](https://www.numpy.org/), [Pandas](https://pandas.pydata.org/), [Scikit Learn](https://scikit-learn.org/stable/), [scipy Sparse](https://pypi.org/project/sparse/0.1.1/), [Surpise](http://surpriselib.com/), [Implicit](https://github.com/benfred/implicit), [LightFM](https://github.com/lyst/lightfm)

```
$ git clone https://github.com/celerative/PPS-RecSys-Nicolas.git
$ pip install -e .
```

## Features

### Models
#### Recomendation Model
Alternating Least Squares (ALS)
#### Prediction Model
Bayesian Personalized Ranking (BPR)
Global Average
Item Average
User Average
K-Nearest Neighbors (KNN)
* KNN Basic
* KNN Baseline
* KNN with Means
* KNN with Zero Score
Single Value Descomposition (SVD)
Weighted Approximate-Rank Pairwise (WARP)

### Accuracy
#### For Regression Model
Area Under the Curve (AUC)
Mean Absolute Error (MAE)
Root Mean Square Error (RMSE)
#### For Classification Model
Precision
Recall
F1

### Metrics
Cross Validation
Grid Search CV

### Model Selection
Cosine Similarity
Pearson Correlation

### Persistence
Load: load model
Dump: save model

## Support
- Marcos, Federico
- Palazzesi, Nicolás