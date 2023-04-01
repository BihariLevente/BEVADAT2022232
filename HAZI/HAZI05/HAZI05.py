import pandas as pd
import numpy as np
import seaborn as sns
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix

class KNNClassifier:
    def __init__ (self, k : int, test_split_ratio : float) -> None:
        self.k = k
        self.test_split_ratio = test_split_ratio
    
    @staticmethod
    def load_csv(csv_path:str) -> Tuple[pd.DataFrame , pd.Series]:
        dataset = pd.read_csv(csv_path, delimiter=',')
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        x,y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
        return x,y

    def train_test_split(self, features: pd.DataFrame, labels: pd.Series):
        test_size = int(len(features) * self.test_split_ratio)
        train_size = len(features) - test_size

        assert len(features) == test_size + train_size, "Size mismatch!"

        self.x_train = features.iloc[:train_size,:].reset_index(drop=True)
        self.y_train = labels.iloc[:train_size].reset_index(drop=True)
        self.x_test = features.iloc[train_size:train_size+test_size,:].reset_index(drop=True)
        self.y_test = labels.iloc[train_size:train_size + test_size].reset_index(drop=True)
    
    def euclidean(self, element_of_x: pd.Series) -> pd.Series:
        return ((self.x_train - element_of_x)**2).sum(axis=1).apply(np.sqrt)

    def predict(self, x_test: pd.DataFrame):
        labels_pred = []
        for x_test_element in x_test.values:
            distances = self.euclidean(x_test_element)
            distances_df = pd.DataFrame(zip(distances,self.y_train)).sort_values(by='distance')
            label_pred = mode(distances_df[:self.k,1],keepdims=False).mode
            labels_pred.append(label_pred)
        self.y_preds = pd.DataFrame(labels_pred,dtype=pd.int32)
    
    def accuracy(self) -> float:
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test) * 100
    
    def confusion_matrix(self) -> np.ndarray:
        conf_matrix = confusion_matrix(self.y_test,self.y_preds)
        return conf_matrix
    
    def best_k(self) -> Tuple[int, float]:
        best_accuracy_idx = (0, -float("inf"))
        self.k = 1
        while self.k <= 20:
            self.predict(self.x_test)
            current_accuracy = self.accuracy()
            if best_accuracy_idx[1] < current_accuracy:
                best_accuracy_idx[0] = self.k
                best_accuracy_idx[1] = round(current_accuracy,2)
            self.k += 1
        return best_accuracy_idx

    @property
    def k_neighbors(self):
        return self.k