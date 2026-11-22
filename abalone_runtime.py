import numpy as np
import pandas as pd

FEATURES = ["Sex", "Length", "Diameter", "Height", "Whole_weight",
            "Shucked_weight", "Viscera_weight", "Shell_weight"]
NUMERIC = FEATURES[1:]
LIMITS = {"Length": (0.05, 0.90), "Diameter": (0.05, 0.70), "Height": (0.01, 0.40),
          "Whole_weight": (0.002, 3.0), "Shucked_weight": (0.001, 1.6),
          "Viscera_weight": (0.0005, 0.8), "Shell_weight": (0.002, 1.1)}
DEFAULTS = {"Length": 0.545, "Diameter": 0.425, "Height": 0.14, "Whole_weight": 0.80,
            "Shucked_weight": 0.34, "Viscera_weight": 0.17, "Shell_weight": 0.23}


def add_features(X):
    X = X.copy()
    X["Volume"] = X["Length"] * X["Diameter"] * X["Height"]
    X["Shell_ratio"] = X["Shell_weight"] / X["Whole_weight"]
    X["Meat_ratio"] = X["Shucked_weight"] / X["Whole_weight"]
    return X


def prepare(frame, factor=1.0, single=False):
    """Validate user input and convert it to dataset units (value / factor). Returns (data, errors)."""
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        return None, ["Missing columns: " + ", ".join(missing)]
    data = frame[FEATURES].copy()
    data["Sex"] = data["Sex"].astype(str).str.strip().str.upper()
    num = data[NUMERIC].apply(pd.to_numeric, errors="coerce") / factor
    where = lambda m: "" if single else " (row %s)" % ", ".join(str(i + 2) for i in data.index[m][:5])
    errors = []
    m = num.isna().any(axis=1)
    if m.any():
        errors.append("Empty or non-numeric values" + where(m) + ".")
    m = ~data["Sex"].isin(["M", "F", "I"])
    if m.any():
        errors.append("Sex must be M, F or I" + where(m) + ".")
    for c, (lo, hi) in LIMITS.items():
        m = (num[c] < lo) | (num[c] > hi)
        if m.any():
            errors.append("%s must be between %.4g and %.4g%s." % (c, lo * factor, hi * factor, where(m)))
    m = num[["Shucked_weight", "Viscera_weight", "Shell_weight"]].gt(num["Whole_weight"], axis=0).any(axis=1)
    if m.any():
        errors.append("Shucked, viscera or shell weight cannot exceed the whole weight" + where(m) + ".")
    data[NUMERIC] = num
    return data, errors


class NumpyModel:
    def __init__(self, path):
        z = np.load(path)
        self.mean, self.scale = z["mean"], z["scale"]
        self.left, self.right, self.feat = z["left"], z["right"], z["feat"]
        self.thr, self.val = z["thr"], z["val"]
        self.depth = int(z["depth"])

    def _design(self, frame):
        X = add_features(frame[FEATURES])
        num = X[NUMERIC + ["Volume", "Shell_ratio", "Meat_ratio"]].to_numpy(float)
        sex = X["Sex"].astype(str).str.strip().str.upper()
        return np.column_stack([(num - self.mean) / self.scale, (sex == "I").to_numpy(float), (sex == "M").to_numpy(float)])

    def predict(self, frame):
        X = self._design(frame).astype(np.float32)
        n, T = len(X), self.left.shape[0]
        tree = np.arange(T)[:, None]
        rows = np.arange(n)[None, :]
        idx = np.zeros((T, n), dtype=np.int64)
        for _ in range(self.depth + 1):
            lc, rc = self.left[tree, idx], self.right[tree, idx]
            leaf = lc == -1
            x = X[rows, np.where(leaf, 0, self.feat[tree, idx])]
            nxt = np.where(x <= self.thr[tree, idx], lc, rc)
            idx = np.where(leaf, idx, nxt)
        return self.val[tree, idx].mean(axis=0)
