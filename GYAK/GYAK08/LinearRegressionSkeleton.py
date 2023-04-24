import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class LinearRegression:
    def __init__(self, epochs: int = 1000, lr: float = 1e-3):
        self.epochs = epochs
        self.L = lr
        self.m = 0
        self.c = 0

    def fit(self, X: np.array, y: np.array):
        n = float(len(X)) # Number of elements in X
        self.losses = []
        for i in range(self.epochs): 
            y_pred = self.m * X + self.c  # The current predicted value of Y
            residuals = y_pred - y
            loss = np.sum(residuals ** 2)
            self.losses.append(loss)
            D_m = (-2/n) * sum(X * residuals)  # Derivative wrt m
            D_c = (-2/n) * sum(residuals)  # Derivative wrt c
            self.m = self.m - self.L * D_m  # Update m
            self.c = self.c - self.L * D_c  # Update c

    def predict(self, X):
        return self.m * X + self.c
