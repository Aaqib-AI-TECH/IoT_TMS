import os
import json
import random
import itertools
import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = "dataset.csv"
LABEL_COL = "label"
# Three configurations compared in Table 11. 
FEATURE_SETS = {
    "XAI-TMS-1                               
    "XAI-TMS-2                  
    "XAI-TMS-3:
}
HIDDEN = (64, 32, 64)
N_SPLITS = 5
REPEATS = 1                       # 1 -> 5-fold (N=5); e.g. 10 -> N=50 (recommended for power)
EPOCHS = 50
BATCH = 8192
SEED = 42
OUTDIR = "outputs_stats"

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)


# ----------------------------- pure stats (testable) -----------------------------
def kendalls_w(chi2, n_obs, k_groups):
    """Friedman effect size: W in [0,1]."""
    return chi2 / (n_obs * (k_groups - 1))


def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation for a Wilcoxon signed-rank test (a vs b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def adjust_pvalues(pvals, method="bonferroni"):
    p = np.asarray(pvals, float)
    m = len(p)
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    if method == "holm":
        order = np.argsort(p)
        adj = np.empty(m)
        running = 0.0
        for rank, idx in enumerate(order):
            val = (m - rank) * p[idx]
            running = max(running, val)
            adj[idx] = min(running, 1.0)
        return adj
    raise ValueError(method)


def compare(f1_matrix, names):
    """f1_matrix: (N_obs, k) F1 per CV observation per configuration."""
    f1 = np.asarray(f1_matrix, float)
    n_obs, k = f1.shape
    chi2, p_fried = stats.friedmanchisquare(*[f1[:, j] for j in range(k)])
    W = kendalls_w(chi2, n_obs, k)

    pairs, raw_p, rbc = [], [], []
    for i, j in itertools.combinations(range(k), 2):
        try:
            _, pv = stats.wilcoxon(f1[:, i], f1[:, j])
        except ValueError:                       # all differences zero
            pv = 1.0
        pairs.append((names[i], names[j]))
        raw_p.append(pv)
        rbc.append(rank_biserial(f1[:, i], f1[:, j]))

    return {
        "n_obs": n_obs,
        "friedman": {"chi2": float(chi2), "p": float(p_fried), "kendalls_w": float(W)},
        "pairwise": [
            {"pair": f"{a} vs {b}", "raw_p": float(rp),
             "bonferroni_p": float(bp), "holm_p": float(hp), "rank_biserial": float(r)}
            for (a, b), rp, bp, hp, r in zip(
                pairs, raw_p,
                adjust_pvalues(raw_p, "bonferroni"),
                adjust_pvalues(raw_p, "holm"), rbc)
        ],
    }


# ----------------------------- model (TensorFlow) -----------------------------
def f1_per_fold():
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    import tensorflow as tf
    from tensorflow.keras import layers, models

    df = pd.read_csv(DATA_PATH)
    y_all = df[LABEL_COL].astype(int).values
    names = list(FEATURE_SETS.keys())

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=REPEATS, random_state=SEED)
    folds = list(rskf.split(df, y_all))
    f1_matrix = np.zeros((len(folds), len(names)))

    for fi, (tr, te) in enumerate(folds):
        for ci, name in enumerate(names):
            cols = FEATURE_SETS[name] or [c for c in df.columns if c != LABEL_COL]
            X = df[cols].astype(float).values
            Xtr, Xte, ytr, yte = X[tr], X[te], y_all[tr], y_all[te]
            mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1     # train-only stats (no leakage)
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            tf.random.set_seed(SEED + fi)
            n = Xtr.shape[1]
            inp = layers.Input(shape=(n,))
            h = layers.Dense(HIDDEN[0], activation="relu")(inp)
            h = layers.Dense(HIDDEN[1], activation="relu")(h)
            h = layers.Dense(HIDDEN[2], activation="relu")(h)
            ae = models.Model(inp, layers.Dense(n)(h)); ae.compile(optimizer="adam", loss="mse")
            ae.fit(Xtr, Xtr, epochs=EPOCHS, batch_size=BATCH, verbose=0)
            clf = LogisticRegression(max_iter=200).fit(ae.predict(Xtr, verbose=0), ytr)
            f1_matrix[fi, ci] = f1_score(yte, clf.predict(ae.predict(Xte, verbose=0)))
        print(f"  fold {fi+1}/{len(folds)}: " +
              ", ".join(f"{names[c]}={f1_matrix[fi,c]:.4f}" for c in range(len(names))))
    return f1_matrix, names


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[config] {N_SPLITS}-fold x {REPEATS} repeat(s) -> N={N_SPLITS*REPEATS} observations")
    f1_matrix, names = f1_per_fold()
    res = compare(f1_matrix, names)

    print(f"\n=== Results (N = {res['n_obs']}) ===")
    fr = res["friedman"]
    print(f"Friedman: chi2={fr['chi2']:.3f}, p={fr['p']:.4f}, Kendall's W={fr['kendalls_w']:.3f}")
    print("Pairwise Wilcoxon (raw / Bonferroni / Holm | rank-biserial r):")
    for row in res["pairwise"]:
        print(f"  {row['pair']:40s} {row['raw_p']:.4f} / {row['bonferroni_p']:.4f} / "
              f"{row['holm_p']:.4f} | r={row['rank_biserial']:+.3f}")
    mean_f1 = f1_matrix.mean(0); sd_f1 = f1_matrix.std(0, ddof=1)
    res["mean_f1"] = {n: [float(mean_f1[i]), float(sd_f1[i])] for i, n in enumerate(names)}
    json.dump(res, open(f"{OUTDIR}/stats.json", "w"), indent=2)
    print(f"\nSaved {OUTDIR}/stats.json")


if __name__ == "__main__":
    main()
