import os
import json
import random
import numpy as np
import pandas as pd

DATA_PATH = "dataset.csv"
LABEL_COL = "label"
HIDDEN = (64, 32, 64)
NORM = "minmax"                   
TEST_SIZE = 0.30
EPOCHS = 100
BATCH = 8192
N_RUNS = 5
BASE_SEED = 42
OUTDIR = "outputs_configB"

os.environ["PYTHONHASHSEED"] = str(BASE_SEED)
random.seed(BASE_SEED); np.random.seed(BASE_SEED)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def scale(train, *others):
    if NORM == "zscore":
        c, s = train.mean(), train.std().replace(0, 1)
        return [(train - c) / s] + [(o - c) / s for o in others]
    lo = train.min(); rng = (train.max() - lo).replace(0, 1)   # fit on training only
    return [(train - lo) / rng] + [(o - lo) / rng for o in others]


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    dr = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * dr / (prec + dr) if (prec + dr) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    return dict(DR=dr, FAR=far, precision=prec, F1=f1, accuracy=acc)


def run_once(df, seed):
    set_seed(seed)
    import tensorflow as tf
    from tensorflow.keras import layers, models
    cols = SELECTED_FEATURES if SELECTED_FEATURES else [c for c in df.columns if c != LABEL_COL]
    X, y = df[cols].astype(float), df[LABEL_COL].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
    med = Xtr.median(); Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    Xtr_s, Xte_s = scale(Xtr, Xte)

    n = Xtr_s.shape[1]
    inp = layers.Input(shape=(n,))
    h = layers.Dense(HIDDEN[0], activation="relu")(inp)
    h = layers.Dense(HIDDEN[1], activation="relu")(h)
    h = layers.Dense(HIDDEN[2], activation="relu")(h)
    out = layers.Dense(n, activation="linear")(h)
    ae = models.Model(inp, out); ae.compile(optimizer="adam", loss="mse")
    ae.fit(Xtr_s.values, Xtr_s.values, epochs=EPOCHS, batch_size=BATCH, verbose=0)

    feat_tr = ae.predict(Xtr_s.values, verbose=0)
    feat_te = ae.predict(Xte_s.values, verbose=0)
    clf = LogisticRegression(max_iter=200).fit(feat_tr, ytr)
    m = metrics(yte.values, clf.predict(feat_te))
    m["seed"] = seed
    return m


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print(f"[config] NORM={NORM} | features={len(SELECTED_FEATURES) if SELECTED_FEATURES else 'all'} "
          f"| n_runs={N_RUNS}")
    runs = []
    for i in range(N_RUNS):
        m = run_once(df, BASE_SEED + i)
        runs.append(m)
        print(f"[run {i+1}] DR={m['DR']:.4f} FAR={m['FAR']:.4f} F1={m['F1']:.4f} acc={m['accuracy']:.4f}")
    keys = ["DR", "FAR", "precision", "F1", "accuracy"]
    print(f"\n=== Config B (supervised) over {N_RUNS} seeds (mean +/- 95% CI) ===")
    ci = {}
    for k in keys:
        v = np.array([r[k] for r in runs], float)
        half = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        ci[k] = [float(v.mean()), float(v.mean() - half), float(v.mean() + half)]
        print(f"  {k:9s}: {v.mean():.4f}  (95% CI {ci[k][1]:.4f}, {ci[k][2]:.4f})")
    json.dump({"runs": runs, "ci": ci}, open(f"{OUTDIR}/metrics_configB.json", "w"), indent=2)
    print(f"\nSaved {OUTDIR}/metrics_configB.json")


if __name__ == "__main__":
    main()
