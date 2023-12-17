import numpy as np


def covariance_matrix(X, Y=None, rowvar=False):
    """Returns the covariance matrix of X"""
    if Y is None:
        Y = X
    if rowvar:
        X = X.T
        Y = Y.T
    mean_x = np.mean(X, axis=0)
    mean_y = np.mean(Y, axis=0)
    return (X - mean_x) @ (Y - mean_y).T / (X.shape[0] - 1)
