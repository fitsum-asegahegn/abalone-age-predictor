import sys
import joblib
import numpy as np
import abalone_tools  # noqa: F401

pipe = joblib.load(sys.argv[1] if len(sys.argv) > 1 else "abalone_model.joblib")
rf = pipe.named_steps["model"]
scaler = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
trees = [e.tree_ for e in rf.estimators_]
width = max(t.node_count for t in trees)


def pad(get, fill, dtype):
    out = np.full((len(trees), width), fill, dtype=dtype)
    for i, t in enumerate(trees):
        a = get(t)
        out[i, : len(a)] = a
    return out


np.savez_compressed(
    "abalone_model.npz",
    mean=scaler.mean_, scale=scaler.scale_,
    left=pad(lambda t: t.children_left, -1, np.int64), right=pad(lambda t: t.children_right, -1, np.int64),
    feat=pad(lambda t: t.feature, 0, np.int64), thr=pad(lambda t: t.threshold, 0.0, np.float64),
    val=pad(lambda t: t.value[:, 0, 0], 0.0, np.float64), depth=max(t.max_depth for t in trees),
)
print("saved abalone_model.npz", len(trees), "trees")
