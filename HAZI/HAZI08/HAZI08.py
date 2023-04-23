import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error

def load_iris_data() -> sklearn.utils.Bunch: 
    return load_iris()

def check_data(iris) -> pd.core.frame.DataFrame:
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    return iris_df.head(5)

def linear_train_data(iris) -> (np.ndarray, np.ndarray):
    iris_database = iris
    X = np.array(iris_database.data[:,1:4])
    y = np.array(iris_database.data[:,0])
    return (X,y)

def logistic_train_data(iris) -> (np.ndarray, np.ndarray):
    iris_database = iris
    mask = np.where((iris_database.target == 0) | (iris_database.target == 1))
    X = np.array(iris_database.data[mask])
    y = np.array(iris_database.target[mask])
    return (X,y)

def split_data(X, y) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_linear_regression(X_train, y_train) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train) -> LogisticRegression: #error with "sklearn.linear_model._base.LogisticRegression"
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def predict(model , X_test) -> np.ndarray:
    return model.predict(X_test)

def plot_actual_vs_predicted(y_test, y_pred) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.set_title('Actual vs Predicted Target Values')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.scatter(y_test, y_pred, color='blue')
    min, max = y_test.min(), y_test.max()
    ax.plot([min, max], [min, max], color='red')
    return fig

def evaluate_model(y_test, y_pred) -> float:
    mse = mean_squared_error(y_test,y_pred)
    return mse