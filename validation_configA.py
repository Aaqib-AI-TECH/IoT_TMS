import os
import json
import random
import numpy as np
import pandas as pd

# ----------------------------- configuration -----------------------------
DATA_PATH = "dataset.csv"          
LABEL_COL = "label"
TRUSTWORTHY_LABEL = 0              # 0 = trustworthy, 1 = untrustworthy
TEST_SIZE = 0.30                  # 70/30 train+val / test split
VAL_FRAC = 0.20                   # validation = 20% of the trustworthy training pool
EPOCHS = 100
BATCH = 8192
HIDDEN = (64, 32, 64)             # encoder-bottleneck-decoder widths
PERCENTILE = 99.0                 # detection threshold percentile on validation error
NORM = "minmax"                   # "minmax"
SWEEP_PERCENTILES = [90, 95, 97.5, 99, 99.5, 99.9]
N_RUNS = 5                        # >1 -> reports mean +/- 95% CI; set 1 for a quick run
BASE_SEED = 42
OUTDIR = "outputs_configA"

# Pin the run to a specific feature set so it matches a Table-4 row.
# Set to None to use every column except the label.
SELECTED_FEATURES = []

# numpy / python seeds can be set before any TF import
os.environ["PYTHONHASHSEED"] = str(BASE_SEED)
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score,
)

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------- helpers -----------------------------
def set_global_seed(seed):
    """Make a run reproducible across python, numpy and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()  # TF >= 2.8
    except Exception:
        pass


def build_and_train(X_train_s, X_val_s, seed):
    """Build and train the autoencoder on trustworthy data only. Lazy TF import
    keeps the rest of the pipeline testable without TensorFlow installed."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    n_features = X_train_s.shape[1]
    inp = layers.Input(shape=(n_features,))
    x = layers.Dense(HIDDEN[0], activation="relu")(inp)
    x = layers.Dense(HIDDEN[1], activation="relu")(x)
    x = layers.Dense(HIDDEN[2], activation="relu")(x)
    out = layers.Dense(n_features, activation="linear")(x)
    autoencoder = models.Model(inp, out)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(
        X_train_s.values, X_train_s.values,
        epochs=EPOCHS, batch_size=BATCH,
        validation_data=(X_val_s.values, X_val_s.values),
        verbose=0,
    )
    return autoencoder


def recon_error(model, X_df):
    """Per-sample mean squared reconstruction error (robust to DataFrame input)."""
    recon = model.predict(X_df.values, verbose=0)
    Xv = X_df.values.astype(float)
    return np.mean((Xv - recon) ** 2, axis=1)


def detection_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    dr = tp / (tp + fn) if (tp + fn) else 0.0          # detection rate = recall(untrustworthy)
    far = fp / (fp + tn) if (fp + tn) else 0.0          # false alarm rate
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * dr / (precision + dr)) if (precision + dr) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return dict(DR=dr, FAR=far, precision=precision, recall=dr, F1=f1,
                accuracy=accuracy, TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn))


def run_once(df, seed, build_fn=build_and_train):
    """One full reproducible pass; returns a metrics dict plus arrays for plotting."""
    if build_fn is build_and_train:   # only touch TF seeds when really training
        set_global_seed(seed)

    cols = SELECTED_FEATURES if SELECTED_FEATURES else [c for c in df.columns if c != LABEL_COL]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"These selected features are not in the dataset: {missing}")
    X = df[cols]
    y = df[LABEL_COL].astype(int)

    # 70/30 split, stratified so the test set keeps both classes
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    # autoencoder sees TRUSTWORTHY data only; validation is a held-out slice of it
    X_train = X_trainval[y_trainval == TRUSTWORTHY_LABEL].copy()
    X_val = X_train.sample(frac=VAL_FRAC, random_state=seed)
    X_train = X_train.drop(X_val.index)

    # leakage-free imputation + standardisation: ALL statistics come from X_train only
    med = X_train.median()
    X_train = X_train.fillna(med)
    X_val = X_val.fillna(med)
    X_test = X_test.fillna(med)

    mean = X_train.mean()
    std = X_train.std().replace(0, 1)
    if NORM == "zscore":
        X_train_s = (X_train - mean) / std
        X_val_s = (X_val - mean) / std
        X_test_s = (X_test - mean) / std
    elif NORM == "minmax":
        lo = X_train.min()
        rng = (X_train.max() - lo).replace(0, 1)   # fit on training data only
        X_train_s = (X_train - lo) / rng
        X_val_s = (X_val - lo) / rng
        X_test_s = (X_test - lo) / rng
    else:
        raise ValueError(f"Unknown NORM={NORM!r}; use 'minmax'")

    model = build_fn(X_train_s, X_val_s, seed)

    errs_val = recon_error(model, X_val_s)
    errs_test = recon_error(model, X_test_s)

    threshold = np.percentile(errs_val, PERCENTILE)
    y_pred = (errs_test > threshold).astype(int)

    m = detection_metrics(y_test.values, y_pred)
    m["threshold"] = float(threshold)
    m["ROC_AUC"] = float(roc_auc_score(y_test, errs_test))
    m["PR_AUC"] = float(average_precision_score(y_test, errs_test))
    m["seed"] = seed

    arrays = dict(y_test=y_test.values, errs_test=errs_test,
                  errs_val=errs_val, threshold=threshold)
    return m, arrays


# ----------------------------- reporting -----------------------------
def aggregate_ci(runs, keys):
    """Mean +/- std and normal-approx 95% CI across runs."""
    out = {}
    n = len(runs)
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        mean = vals.mean()
        std = vals.std(ddof=1) if n > 1 else 0.0
        half = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        out[k] = dict(mean=mean, std=std, ci95=(mean - half, mean + half))
    return out


def threshold_sweep(arrays):
    """Per-seed sweep: for each percentile, calibrate tau on this seed's validation
    error and evaluate on this seed's test set."""
    rows = []
    for p in SWEEP_PERCENTILES:
        thr = np.percentile(arrays["errs_val"], p)
        y_pred = (arrays["errs_test"] > thr).astype(int)
        m = detection_metrics(arrays["y_test"], y_pred)
        rows.append(dict(percentile=p, threshold=thr,
                         DR=m["DR"], FAR=m["FAR"], F1=m["F1"]))
    return pd.DataFrame(rows)


def averaged_sweep(all_arrays):
    """Seed-averaged sweep so the selected operating point matches the Table-4 mean.
    Averages DR/FAR/F1 across seeds at each percentile, with 95% CI on DR and FAR."""
    per_seed = [threshold_sweep(a) for a in all_arrays]      # list of DataFrames
    n = len(per_seed)
    rows = []
    for i, p in enumerate(SWEEP_PERCENTILES):
        dr = np.array([s.loc[i, "DR"] for s in per_seed], float)
        far = np.array([s.loc[i, "FAR"] for s in per_seed], float)
        f1 = np.array([s.loc[i, "F1"] for s in per_seed], float)
        def ci(v):
            h = 1.96 * v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            return v.mean(), v.mean() - h, v.mean() + h
        dr_m, dr_lo, dr_hi = ci(dr)
        far_m, far_lo, far_hi = ci(far)
        rows.append(dict(percentile=p,
                         DR=dr_m, DR_lo=dr_lo, DR_hi=dr_hi,
                         FAR=far_m, FAR_lo=far_lo, FAR_hi=far_hi,
                         F1=f1.mean()))
    return pd.DataFrame(rows)


def make_plots(arrays, outdir):
    y_test, errs_test = arrays["y_test"], arrays["errs_test"]
    thr = arrays["threshold"]

    # ROC
    fpr, tpr, _ = roc_curve(y_test, errs_test)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, errs_test):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False alarm rate"); plt.ylabel("Detection rate")
    plt.title("ROC - Config A (unsupervised)"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/roc.png", dpi=200); plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_test, errs_test)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, label=f"AP = {average_precision_score(y_test, errs_test):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall - Config A"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/pr.png", dpi=200); plt.close()

    # reconstruction-error distribution with threshold line
    plt.figure(figsize=(6, 4))
    bins = np.linspace(0, np.percentile(errs_test, 99.5), 60)
    plt.hist(errs_test[y_test == 0], bins=bins, alpha=0.6, label="Trustworthy", density=True)
    plt.hist(errs_test[y_test == 1], bins=bins, alpha=0.6, label="Untrustworthy", density=True)
    plt.axvline(thr, color="red", linestyle="--", label=f"Threshold ({PERCENTILE:.1f}th pct)")
    plt.xlabel("Reconstruction error (MSE)"); plt.ylabel("Density")
    plt.title("Reconstruction error distribution"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/error_distribution.png", dpi=200); plt.close()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    feat = "selected (%d)" % len(SELECTED_FEATURES) if SELECTED_FEATURES else "all columns"
    print(f"[config] NORM={NORM} | features={feat} | percentile={PERCENTILE} | "
          f"epochs={EPOCHS} | n_runs={N_RUNS}\n")

    runs, first_arrays, all_arrays = [], None, []
    for i in range(N_RUNS):
        seed = BASE_SEED + i
        m, arrays = run_once(df, seed)
        runs.append(m)
        all_arrays.append(arrays)
        if first_arrays is None:
            first_arrays = arrays
        print(f"[run {i+1}/{N_RUNS} seed={seed}] "
              f"DR={m['DR']:.4f} FAR={m['FAR']:.4f} F1={m['F1']:.4f} "
              f"ROC_AUC={m['ROC_AUC']:.4f} thr={m['threshold']:.5f}")

    # canonical single-run report (first seed)
    y_pred0 = (first_arrays["errs_test"] > first_arrays["threshold"]).astype(int)
    print("\nClassification report (seed={}):".format(BASE_SEED))
    print(classification_report(first_arrays["y_test"], y_pred0, digits=4, zero_division=0))

    # confidence intervals
    keys = ["DR", "FAR", "precision", "recall", "F1", "accuracy", "ROC_AUC", "PR_AUC"]
    ci = aggregate_ci(runs, keys)
    print(f"\n=== Config A metrics over {N_RUNS} seed(s) (mean +/- 95% CI) ===")
    for k in keys:
        c = ci[k]
        print(f"{k:10s}: {c['mean']:.4f}  (95% CI {c['ci95'][0]:.4f}, {c['ci95'][1]:.4f})")

    # threshold sensitivity (averaged over all seeds, so it matches Table 4)
    sweep = averaged_sweep(all_arrays)
    print(f"\n=== Threshold sensitivity, mean over {N_RUNS} seed(s) "
          f"(validation percentile -> test DR/FAR) ===")
    show = sweep.copy()
    for c in ["DR", "DR_lo", "DR_hi", "FAR", "FAR_lo", "FAR_hi", "F1"]:
        show[c] = show[c].round(4)
    print(show.to_string(index=False))
    sweep.to_csv(f"{OUTDIR}/threshold_sweep.csv", index=False)

    # sweep plot with 95% CI bands
    plt.figure(figsize=(6, 4))
    plt.plot(sweep["percentile"], sweep["DR"], "o-", label="DR")
    plt.fill_between(sweep["percentile"], sweep["DR_lo"], sweep["DR_hi"], alpha=0.2)
    plt.plot(sweep["percentile"], sweep["FAR"], "s-", label="FAR")
    plt.fill_between(sweep["percentile"], sweep["FAR_lo"], sweep["FAR_hi"], alpha=0.2)
    plt.axvline(PERCENTILE, color="grey", linestyle="--", label=f"selected ({PERCENTILE:.1f}th)")
    plt.xlabel("Validation-error percentile (threshold)"); plt.ylabel("Rate")
    plt.title(f"Threshold sensitivity - Config A (mean of {N_RUNS} seeds)"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/threshold_sensitivity.png", dpi=200); plt.close()

    make_plots(first_arrays, OUTDIR)

    # persist machine-readable results
    with open(f"{OUTDIR}/metrics.json", "w") as f:
        json.dump({"runs": runs, "ci": ci, "config": {
            "PERCENTILE": PERCENTILE, "EPOCHS": EPOCHS, "BATCH": BATCH,
            "HIDDEN": HIDDEN, "N_RUNS": N_RUNS, "BASE_SEED": BASE_SEED}}, f, indent=2)
    print(f"\nSaved figures, threshold_sweep.csv and metrics.json to ./{OUTDIR}/")


if __name__ == "__main__":
    main()
