import pandas as pd

from pyrecsys.models.knn import KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore

def test_knn():
    #Read File
    file_path = 'ratings_small.csv'
    df = pd.read_csv(file_path, dtype='unicode')

    #Change type of data
    df.userId = df.userId.astype(int)
    df.movieId = df.movieId.astype(int)
    df.rating = df.rating.astype(float)

    #Sort Values
    df.sort_values(by=['userId','movieId'],ascending=True)

    X = df[['userId','movieId']].values
    y = df.rating.values

    # Create KNN model
    model = KNNBasic()
    model2 = KNNBaseline()
    model3 = KNNWithMeans()
    model4 = KNNWithZScore()

    # Fit model
    model.fit(X,y)
    model2.fit(X,y)
    model3.fit(X,y)
    model4.fit(X,y)

    # Testing model 
    #result = model.predict(X)
    uid = 1
    N = 10
    result = model.recommend(uid,N)
    assert result.shape[0] == N
    result2 = model2.recommend(uid,N)
    assert result2.shape[0] == N
    result3 = model3.recommend(uid,N)
    assert result3.shape[0] == N
    result4 = model4.recommend(uid,N)
    assert result4.shape[0] == N
    