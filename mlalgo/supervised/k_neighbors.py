import numpy as np


class KNeighborsClassifier:

    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        closest = self._get_closest(X)
        return self._get_predictions(closest)
    
    def _get_closest(self, X):
        distances = np.linalg.norm(self.X[:, np.newaxis] - X, axis=2)
        closest = np.argsort(distances, axis=0)[:self.n_neighbors]
        return closest
    
    def _get_predictions(self, closest):
        classes = self.y[closest]
        class_ = np.apply_along_axis(self.most_frequent_in_column, 0, classes)
        return class_
    
    @staticmethod
    def most_frequent_in_column(arr):
        return np.argmax(np.bincount(arr)) 
