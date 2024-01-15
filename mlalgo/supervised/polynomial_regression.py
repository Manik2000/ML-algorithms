from itertools import combinations, combinations_with_replacement
from math import comb

import numpy as np

from .linear_regression import LinearRegression


class PolynomialRegression:
    def __init__(self, degree, interaction_only=False):
        self.degree = degree
        self.interaction_only = interaction_only
        self.theta = None

    def fit(self, X, Y):
        extended_X = self.create_polynomial_features(X)
        self.theta = LinearRegression(fit_intercept=False).fit(extended_X, Y).theta
        return self

    def predict(self, X):
        extended_X = self.create_polynomial_features(X)
        return extended_X @ self.theta

    def create_polynomial_features(self, X):
        number_of_features = X.shape[1]
        features_nums = range(number_of_features)
        if self.interaction_only:
            m = sum(
                comb(number_of_features, i)
                for i in range(min(self.degree + 1, number_of_features + 1))
            )
        else:
            m = sum(comb(number_of_features + i - 1, i) for i in range(self.degree + 1))
        extended_X = np.zeros((X.shape[0], m))
        extended_X[:, 0] = 1
        idx = 1
        if self.interaction_only:
            for i in range(1, self.degree + 1):
                for combinantion in combinations(features_nums, i):
                    extended_X[:, idx] = np.prod(X[:, combinantion], axis=1)
                    idx += 1
        else:
            for i in range(1, self.degree + 1):
                for combinantion in combinations_with_replacement(features_nums, i):
                    extended_X[:, idx] = np.prod(X[:, combinantion], axis=1)
                    idx += 1
        return extended_X
