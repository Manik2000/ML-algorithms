import numpy as np


class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.theta = None

    def fit(self, X, Y):
        """
        Fit the model to data matrix X and target vector Y.
        Use explicit formula for theta.

        X - np.array, shape = (n, m)
        Y - np.array, shape = (n)
        """
        if self.fit_intercept:
            self.extended_X = np.hstack([np.ones((len(X), 1)), X])
        else:
            self.extended_X = X
        self.theta = np.linalg.solve(
            self.extended_X.T @ self.extended_X, self.extended_X.T @ Y
        )
        return self

    def predict(self, X):
        """
        Return the model's prediction for data matrix X.

        X - np.array, shape = (k, m)
        """
        if self.fit_intercept:
            return np.hstack([np.ones((len(X), 1)), X]) @ self.theta
        return X @ self.theta
