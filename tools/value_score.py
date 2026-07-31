"""Learned value function as a drop-in state scorer for build_sft's adoption filter.

Replaces `eval_state.evaluate()` with P(win) from the model trained by value_net.py,
which discriminates winner-from-loser far better exactly where the handcrafted one is
blind: 0-15% 41.2 -> 61.5, 25% 51.2 -> 66.8, 50% 67.8 -> 75.4 (see value-net-v1).

SCALE: scores are **win probability in percentage points** (0-100), so a delta of -1.0
means "this move cost 1pp of win probability". The handcrafted evaluator's scale was
1 prize = 1000 and 1 hand card = 1.0, so `--eval-margin` / `--eval-temp` DO NOT carry
over and must be recalibrated in pp (tools/calibrate_value_margin.py).

ARTIFACT: prefer `value_data.npz` and refit here. A sklearn/numpy pickle written on the
Kaggle image (numpy 2.x) will not unpickle under numpy 1.26, and the fit is under a
minute, so the feature matrix is the portable artifact -- not the model.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                    # noqa: E402
from value_net import featurize                       # noqa: E402


def _fit_from_npz(path):
    from sklearn.ensemble import HistGradientBoostingClassifier
    d = np.load(path, allow_pickle=False)
    X, y = d["X"], d["y"]
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
        min_samples_leaf=100, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=0)
    clf.fit(X, y)
    return clf


class ValueScorer:
    """Callable (state, me) -> win probability in percentage points."""

    def __init__(self, path):
        if os.path.isdir(path):
            npz = os.path.join(path, "value_data.npz")
            pkl = os.path.join(path, "value_model.pkl")
        else:
            npz, pkl = (path, "") if path.endswith(".npz") else ("", path)
        if npz and os.path.exists(npz):
            self.model = _fit_from_npz(npz)
            self.source = f"refit({os.path.basename(npz)})"
        elif pkl and os.path.exists(pkl):
            import pickle
            self.model = pickle.load(open(pkl, "rb"))["model"]
            self.source = f"pickle({os.path.basename(pkl)})"
        else:
            raise FileNotFoundError(f"no value artifact at {path}")

    def __call__(self, st, me):
        f = featurize(st, me)
        if f is None:
            return None
        return float(self.model.predict_proba(np.asarray([f], dtype=np.float32))[0, 1]) * 100.0

    def batch_pairs(self, items):
        """Score MANY (key, S0, S1) triples in ONE predict_proba call.

        `pair()` is two rows per call, and sklearn's per-call overhead dominates at that
        size: measured 1.77 ms/state at batch 2 vs 0.021 ms/state at batch 300 -- an 82x
        difference that is pure call overhead, not model cost. build_sft used to call
        pair() once per MAIN candidate, so the learned filter ran 146x slower than the
        handcrafted evaluate() it replaced (0.045 ms -> 6.64 ms) and turned a ~20 min
        Kaggle build into one that never finished inside the 12 h cap. Batching a whole
        GAME's candidates restores it: featurize (0.094 ms/state) becomes the floor.

        Returns {key: (e0, e1)}, either element None when that state can't be featurized.
        """
        feats, slots = [], []
        for key, me, st0, st1 in items:
            f0 = featurize(st0, me) if st0 is not None else None
            f1 = featurize(st1, me) if st1 is not None else None
            i0 = i1 = None
            if f0 is not None:
                i0 = len(feats); feats.append(f0)
            if f1 is not None:
                i1 = len(feats); feats.append(f1)
            slots.append((key, i0, i1))
        if not feats:
            return {key: (None, None) for key, _, _ in slots}
        p = self.model.predict_proba(np.asarray(feats, dtype=np.float32))[:, 1] * 100.0
        return {key: (None if i0 is None else float(p[i0]),
                      None if i1 is None else float(p[i1])) for key, i0, i1 in slots}

    def pair(self, st0, st1, me):
        """Score two states in ONE predict call -- build_sft always wants (S0, S1).

        Kept for callers outside the build loop; build_sft uses batch_pairs(), which is
        ~24x faster end-to-end. Do NOT call this per candidate in a bulk job."""
        f0 = featurize(st0, me) if st0 is not None else None
        f1 = featurize(st1, me) if st1 is not None else None
        if f0 is None or f1 is None:
            return (None if f0 is None else self(st0, me),
                    None if f1 is None else self(st1, me))
        p = self.model.predict_proba(np.asarray([f0, f1], dtype=np.float32))[:, 1]
        return float(p[0]) * 100.0, float(p[1]) * 100.0
