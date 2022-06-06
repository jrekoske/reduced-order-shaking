from sklearn.base import BaseEstimator, RegressorMixin
from scipy.interpolate import RBFInterpolator


class RBFRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, kernel=None, smoothing=None):
        self.kernel = kernel
        self.smoothing = smoothing

    def fit(self, X, y):
        self.rbf = RBFInterpolator(
            X, y, kernel=self.kernel, smoothing=self.smoothing)
        return self

    def predict(self, X):
        return self.rbf(X)
