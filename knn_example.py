import pandas as pd
from sklearn.model_selection import train_test_split

from models.knn import KNN

#read in the data using pandas
df = pd.read_csv('diabetes.csv')

#create a dataframe with all training data except the target column
X = df.drop(columns=['Outcome'])

#separate target values
y = df['Outcome'].values

#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Create KNN model
model = KNN(n_neighbors = 3)

# Fit the classifier to the data
model.fit(X_train,y_train)

# Testing model - Show first 5 model predictions on the test data
array_result = model.predict(X_test)[0:5]
for a in array_result:
    if(a == 0):
        print("The pacient is not diabetic")
    else:
        print("The pacient is diabetic")