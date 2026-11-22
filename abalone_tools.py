import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

from abalone_runtime import FEATURES, NUMERIC, LIMITS, DEFAULTS, add_features, prepare


class VIFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=10.0):
        self.threshold = threshold

    def _vif(self, A):
        out = []
        for j in range(A.shape[1]):
            others = np.delete(A, j, axis=1)
            r2 = LinearRegression().fit(others, A[:, j]).score(others, A[:, j])
            out.append(1.0 / max(1.0 - r2, 1e-12))
        return np.array(out)

    def fit(self, X, y=None):
        A = np.asarray(X, dtype=float)
        keep = list(range(A.shape[1]))
        while len(keep) > 2:
            v = self._vif(A[:, keep])
            if v.max() < self.threshold:
                break
            keep.pop(int(v.argmax()))
        self.keep_ = np.array(keep)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]
