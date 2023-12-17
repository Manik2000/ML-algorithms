import numpy as np

from ..utils.helper_functions import covariance_matrix 


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.X_centered = X - self.mean
        cov = np.cov(self.X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        idx = eigenvalues.argsort()[::-1]
        self.components_ = eigenvectors[:, idx][:, : self.n_components]
        self.explained_variance_ = eigenvalues[idx][: self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvalues)
        return self

    def transform(self, X):
        X_centered = X - np.mean(X, axis=0)
        return X_centered @ self.components_

    def fit_transform(self, X):
        self.fit(X)
        self.transform(X)
