import numpy as np


class KMeans:

    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            clusters = self._assign_clusters(X)
            self._update_centroids(X, clusters)
        return self

    def predict(self, X):
        return self._assign_clusters(X)
    
    def fit_predict(self, X):
        return self.fit(X).predict(X)
    
    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, clusters):
        for i in range(self.n_clusters):
            self.centroids[i] = np.mean(X[clusters == i], axis=0)
