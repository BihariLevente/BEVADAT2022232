import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from sklearn import datasets

class KMeansOnDigits():
    def __init__(self, n_clusters, random_state):
        self.n_clusters = n_clusters 
        self.random_state = random_state

    def load_digits(self):
        self.digits =  datasets.load_digits()

    def predict(self):
        kmeans = KMeans(n_clusters = self.n_clusters, random_state = self.random_state)
        self.clusters = kmeans.fit_predict(self.digits.data)

    def get_labels(self) -> np.ndarray:
        result_array = np.zeros_like(self.clusters)
        #result_array = np.array([None] * len(clusters))
        for i in range(self.digits.target_names.shape[0]):
            mask = self.clusters == i
            sub_array = self.digits.target[mask]
            mode_element = mode(sub_array)
            result_array[mask] = mode_element[0]

        self.labels = result_array

    def calc_accuracy(self):
        self.accuracy = np.round(accuracy_score(self.digits.target, self.labels),2)

    def confusion_matrix(self):
        self.mat = confusion_matrix(self.digits.target, self.labels)