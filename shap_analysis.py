"""
SHAP feature attribution for XAI-TMS (explains the unsupervised anomaly score).
  * Background-size sensitivity: repeats the attribution with |background| in {50, 100, 200} and
    reports the stability of the top-k feature ranking (Jaccard overlap + Spearman correlation).
  * Outputs the mean |SHAP| ranking used to select the XAI-TMS-3 feature subset.

Reproducible (fixed seeds). TensorFlow and shap are imported lazily.
"""

import os
import json
import random
import numpy as np
import pandas as pd

# ----------------------------- configuration -----------------------------
DATA_PATH = "dataset.csv"
LABEL_COL = "label"
TRUSTWORTHY_LABEL = 0
SELECTED_FEATURES = []         
HIDDEN = (64, 32, 64)
BACKGROUND_SIZES = [50, 100, 200]
N_EXPLAIN = 100                   # untrustworthy instances to attribute
TOP_K = 12                        
SEED = 42
OUTDIR = "outputs_shap"

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)


# ----------------------------- pure-NumPy helpers (testable) -----------------------------
def numpy_forward(weights, X):
    h = X
    for i, (W, b) in enumerate(weights):
        h = h @ W + b
        if i < len(weights) - 1:
            h = np.maximum(h, 0.0)
    return h


def recon_error_np(weights, X):
    X = np.atleast_2d(X).astype(np.float32)
    return np.mean((X - numpy_forward(weights, X)) ** 2, axis=1)


def topk(importances, k):
    return list(np.argsort(importances)[::-1][:k])


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def spearman(x, y):
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else 1.0


# ----------------------------- data + model -----------------------------
def load_scaled():
    df = pd.read_csv(DATA_PATH)
    cols = SELECTED_FEATURES if SELECTED_FEATURES else [c for c in df.columns if c != LABEL_COL]
    X = df[cols].astype(float)
    y = df[LABEL_COL].astype(int).values
    Xtrust = X[y == TRUSTWORTHY_LABEL]
    lo, rng = Xtrust.min(), (Xtrust.max() - Xtrust.min()).replace(0, 1)
    Xs = ((X - lo) / rng).values.astype(np.float32)
    return Xs, y, cols


def build_ae(n_features, Xs, y, seed=SEED):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    tf.random.set_seed(seed)
    inp = layers.Input(shape=(n_features,))
    h = layers.Dense(HIDDEN[0], activation="relu")(inp)
    h = layers.Dense(HIDDEN[1], activation="relu")(h)
    h = layers.Dense(HIDDEN[2], activation="relu")(h)
    out = layers.Dense(n_features, activation="linear")(h)
    ae = models.Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")
    Xtr = Xs[y == TRUSTWORTHY_LABEL]
    ae.fit(Xtr, Xtr, epochs=100, batch_size=8192, verbose=0)
    return ae


# ----------------------------- main -----------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    import shap
    Xs, y, cols = load_scaled()
    n_features = len(cols)
    ae = build_ae(n_features, Xs, y)

    def score(X):                              # anomaly score = reconstruction error
        X = np.atleast_2d(X).astype(np.float32)
        return np.mean((X - ae.predict(X, verbose=0)) ** 2, axis=1)

    rng = np.random.default_rng(SEED)
    trust_idx = np.where(y == TRUSTWORTHY_LABEL)[0]
    untrust_idx = np.where(y == 1)[0]
    explain = Xs[rng.choice(untrust_idx, size=min(N_EXPLAIN, len(untrust_idx)), replace=False)]

    rankings, mean_abs = {}, {}
    for m in BACKGROUND_SIZES:
        bg = Xs[rng.choice(trust_idx, size=m, replace=False)]      # trustworthy background (no leakage)
        explainer = shap.KernelExplainer(score, bg)
        sv = np.array(explainer.shap_values(explain, silent=True))
        imp = np.abs(sv).mean(axis=0)                              # mean |SHAP| per feature
        mean_abs[m] = imp
        rankings[m] = topk(imp, TOP_K)
        print(f"[background={m}] top-{TOP_K} features: {[cols[i] for i in rankings[m]]}")

    # stability across background sizes
    ref = BACKGROUND_SIZES[1] if len(BACKGROUND_SIZES) > 1 else BACKGROUND_SIZES[0]
    print("\n=== Background-size sensitivity (vs background={}) ===".format(ref))
    stability = {}
    for m in BACKGROUND_SIZES:
        j = jaccard(rankings[m], rankings[ref])
        s = spearman(mean_abs[m], mean_abs[ref])
        stability[m] = {"jaccard_topk": round(j, 3), "spearman_all": round(s, 3)}
        print(f"  background={m}: Jaccard(top-{TOP_K})={j:.3f}, Spearman(all)={s:.3f}")

    # final ranking from the reference background
    order = topk(mean_abs[ref], n_features)
    ranking_named = [(cols[i], float(mean_abs[ref][i])) for i in order]
    print(f"\n=== Mean |SHAP| ranking (background={ref}) ===")
    for name, val in ranking_named[:TOP_K]:
        print(f"  {name:18s}: {val:.5f}")

    with open(f"{OUTDIR}/shap_summary.json", "w") as f:
        json.dump({"ranking": ranking_named,
                   "topk_features": [cols[i] for i in rankings[ref]],
                   "stability": stability,
                   "background_sizes": BACKGROUND_SIZES, "n_explain": int(len(explain))}, f, indent=2)
    print(f"\nSaved {OUTDIR}/shap_summary.json")


if __name__ == "__main__":
    main()
