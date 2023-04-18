import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt


class LinearRegression:
    def __init__(self, epochs: int = 2000, lr: float = 1e-3):
        self.epochs = epochs
        self.learning_rate = lr

    def load_data(self,  feature: str, label: str):
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        X,y = df[feature].values, df[label].values
        return X,y

    def fit(self, X: np.array, y: np.array):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.m = 0
        self.c = 0
        n = float(len(self.X_train)) # Number of elements in X
        # Performing Gradient Descent 

        self.losses = []
        for i in range(self.epochs): 
            y_pred = self.m*self.X_train + self.c  # The current predicted value of Y

            residuals = y_pred - self.y_train
            loss = np.sum(residuals ** 2)
            self.losses.append(loss)
            D_m = (-2/n) * sum(self.X_train * residuals)  # Derivative wrt m
            D_c = (-2/n) * sum(residuals)  # Derivative wrt c
            self.m = self.m + self.learning_rate * D_m  # Update m
            self.c = self.c + self.learning_rate * D_c  # Update c
            if i % 100 == 0:
                #print(np.mean(self.y_train-y_pred))
                print(loss)

    def predict(self, X):
        pred = []
        for x in X:
            y_pred = self.m*x + self.c
            pred.append(y_pred)
        
        return pred
