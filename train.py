# Author: Dr. Göktuğ Aşcı
# Creation Date: 06.06.2021
# Licence: MIT license 
# Data: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

# for more explanation regarding the data and models please read knn.ipynb

# to serve the best performing model
# mlflow models serve -m mlruns/0/7375ebe3ad1d4dba9149cb3435f2d1ec/artifacts/model -p 1234

#Importing required packages.
import mlflow
import pandas as pd
import numpy as np
from scipy import stats

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#Loading dataset
df = pd.read_csv('data/winequality-red.csv')

random_state = 42

def remove_outliers(df):
    #Calculate z-scores of `df`
    z_scores = stats.zscore(df)

    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df = df[filtered_entries]
    return df

#Create a reproducible function for the input data
def apply_feature_engineering_preprocessing(df):

    #Making binary classificaion for the response variable.
    #Dividing wine as good and bad by giving the limit for the quality
    bins = (2, 6.5, 8)
    group_names = ['bad', 'good']
    df['quality'] = pd.cut(df['quality'], bins=bins, labels=group_names)

    #Now lets assign a labels to our quality variable
    label_quality = LabelEncoder()

    #Bad becomes 0 and good becomes 1
    df['quality'] = label_quality.fit_transform(df['quality'])

    df = remove_outliers(df)
    return df


#Apply feature engineering
df = apply_feature_engineering_preprocessing(df)


#Now seperate the dataset as response variable and feature variabes
X = df.drop('quality', axis=1)
y = df['quality']


#Train and Test splitting of data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state)


class Model:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.leaf_size = 10
        self.n_jobs = -1
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def try_different_neighbors(self, neighbor_array):
        """
        This function tries different neighbors on the model
        """

        for n_neighbors in neighbor_array:
            knn = KNeighborsClassifier(
                leaf_size=self.leaf_size, n_neighbors=n_neighbors, n_jobs=self.n_jobs)
            knn.fit(self.X_train, self.y_train)

            # let's use the test set to create predictions
            predictions = knn.predict(X_test)

            # calculating the accuracy score manually
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)

            self.log_mlflow(knn, n_neighbors, accuracy, f1)

    def log_mlflow(self, model, n_neighbors, accuracy, f1):
        """
        This function logs model parameters and metrics to mlflow server
        """

        run_name = "KNN - n:{}".format(n_neighbors)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("n_neighbors", n_neighbors)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, "model")


neighbors = [2, 3, 4, 5, 6, 7, 8, 9, 10]

model = Model(X_train, X_test, y_train, y_test)

model.try_different_neighbors(neighbors)
