import os
import json
import random
import numpy as np
import pandas as pd

# ----------------------------- configuration -----------------------------
DATA_PATH = "dataset.csv"
LABEL_COL = "label"
TRUSTWORTHY_LABEL = 0
IMMUTABLE_FEATURES = ['protocol_m']
HIDDEN = (64, 32, 64)
PERCENTILE = 99.0
P_RECON, Q_FLIP, R_CLOSE = 1.0, 5.0, 0.5  
MARGIN_FRAC = 0.0                          
N_ITERS = 600
LR = 0.05
SPARSITY_TOL = 1e-2
N_CF_SAMPLES = 300
VAL_FRAC = 0.20
SEED = 42
OUTDIR = "outputs_cfg"

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)


# ----------------------------- pure-NumPy core (testable) -----------------------------
def numpy_forward(weights, X):
    h = X
    for i, (W, b) in enumerate(weights):
        h = h @ W + b
        if i < len(weights) - 1:
            h = np.maximum(h, 0.0)
    return h


def recon_error_np(weights, X):
    X = np.atleast_2d(X).astype(np.float32)
    r = numpy_forward(weights, X)
    return np.mean((X - r) ** 2, axis=1)


def project(x, x0, immutable_mask, lo, hi):
    x = np.clip(x, lo, hi)
    x[immutable_mask] = x0[immutable_mask]
    return x


def sparsify(cf, x0, immutable_mask, weights, tau, tol=SPARSITY_TOL):
    """Greedily revert changed mutable features to their original value while validity holds."""
    cf2 = cf.copy()
    changed = [i for i in range(len(cf))
               if (not immutable_mask[i]) and abs(cf2[i] - x0[i]) > tol]
    changed.sort(key=lambda i: abs(cf2[i] - x0[i]))   # try reverting smallest changes first
    for i in changed:
        trial = cf2.copy(); trial[i] = x0[i]
        if recon_error_np(weights, trial)[0] <= tau:
            cf2 = trial
    return cf2


def cf_metrics(x0s, cfs, errs_cf, tau, trust_ref, tol=SPARSITY_TOL):
    x0s, cfs = np.asarray(x0s, float), np.asarray(cfs, float)
    valid = np.asarray(errs_cf, float) <= tau
    out = {"n": int(len(cfs)), "validity": float(valid.mean())}
    if valid.sum() == 0:
        out.update(proximity=float("nan"), sparsity=float("nan"), plausibility=float("nan"))
        return out
    xv, cv = x0s[valid], cfs[valid]
    out["proximity"] = float(np.abs(cv - xv).sum(axis=1).mean())
    out["sparsity"] = float((np.abs(cv - xv) > tol).sum(axis=1).mean())
    d = np.sqrt(((cv[:, None, :] - trust_ref[None, :, :]) ** 2).sum(axis=2))
    out["plausibility"] = float(d.min(axis=1).mean())
    return out


# ----------------------------- data + model -----------------------------
def load_scaled():
    df = pd.read_csv(DATA_PATH)
    cols = SELECTED_FEATURES if SELECTED_FEATURES else [c for c in df.columns if c != LABEL_COL]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    X = df[cols].astype(float)
    y = df[LABEL_COL].astype(int).values
    Xtrust = X[y == TRUSTWORTHY_LABEL]
    lo_raw, hi_raw = Xtrust.min(), Xtrust.max()
    rng = (hi_raw - lo_raw).replace(0, 1)                  # min-max fit on trustworthy data only
    Xs = ((X - lo_raw) / rng).values.astype(np.float32)
    return Xs, y, cols, lo_raw.values, rng.values


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


def extract_weights(model):
    pairs = []
    for layer in model.layers:
        w = layer.get_weights()
        if len(w) == 2:
            pairs.append((w[0].astype(np.float32), w[1].astype(np.float32)))
    return pairs


def generate_cf(ae, x0, tau, immutable_mask, lo, hi):
    """Gradient-based CF search (Adam + GradientTape): drive reconstruction error below tau,
    ramp in proximity after the validity phase, keep the most proximal valid CF."""
    import tensorflow as tf
    x0t = tf.constant(x0[None, :], dtype=tf.float32)
    x = tf.Variable(x0[None, :], dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(LR)
    target = (1.0 - MARGIN_FRAC) * tau

    best_cf, best_prox = None, np.inf
    for it in range(N_ITERS):
        r_eff = R_CLOSE * max(0.0, (it - 0.6 * N_ITERS) / (0.4 * N_ITERS))
        with tf.GradientTape() as tape:
            recon = ae(x)
            e = tf.reduce_mean((x - recon) ** 2)
            plaus = e                                       # plausibility (on-manifold)
            valid_term = tf.nn.relu(e - target)             # validity (below threshold)
            close = tf.reduce_sum(tf.abs(x - x0t))          # proximity (L1)
            loss = P_RECON * plaus + Q_FLIP * valid_term + r_eff * close
        g = tape.gradient(loss, x)
        opt.apply_gradients([(g, x)])
        xp = project(x.numpy()[0], x0, immutable_mask, lo, hi)
        x.assign(xp[None, :])
        e_cf = float(np.mean((xp - ae(xp[None, :]).numpy()[0]) ** 2))
        if e_cf <= tau:
            prox = float(np.abs(xp - x0).sum())
            if prox < best_prox:
                best_prox, best_cf = prox, xp.copy()
    return best_cf if best_cf is not None else x.numpy()[0]


# ----------------------------- main -----------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    Xs, y, cols, lo_raw, rng_raw = load_scaled()
    n_features = len(cols)
    immutable_mask = np.array([c in IMMUTABLE_FEATURES for c in cols])
    print(f"[config] features={n_features} | immutable={IMMUTABLE_FEATURES} | "
          f"weights p,q,r=({P_RECON},{Q_FLIP},{R_CLOSE}) | iters={N_ITERS}")

    ae = build_ae(n_features, Xs, y)
    weights = extract_weights(ae)

    # detection threshold tau = 99th percentile of trustworthy validation reconstruction error
    trust = Xs[y == TRUSTWORTHY_LABEL]
    rng = np.random.default_rng(SEED)
    val_idx = rng.choice(len(trust), size=int(VAL_FRAC * len(trust)), replace=False)
    tau = float(np.percentile(recon_error_np(weights, trust[val_idx]), PERCENTILE))
    print(f"detection threshold tau (P{PERCENTILE:.0f} of validation error) = {tau:.6f}")

    lo = np.percentile(trust, 1, axis=0)
    hi = np.percentile(trust, 99, axis=0)
    trust_ref = trust[rng.choice(len(trust), size=min(2000, len(trust)), replace=False)]

    # candidates: untrustworthy instances the detector actually flags (error > tau)
    errs_all = recon_error_np(weights, Xs)
    idx = np.where((y == 1) & (errs_all > tau))[0]
    idx = idx[:N_CF_SAMPLES]
    print(f"Generating counterfactuals for {len(idx)} flagged untrustworthy instances...")

    x0s, cfs = [], []
    for j in idx:
        cf = generate_cf(ae, Xs[j], tau, immutable_mask, lo, hi)
        cf = sparsify(cf, Xs[j], immutable_mask, weights, tau)   # minimise # changed features
        x0s.append(Xs[j]); cfs.append(cf)
    x0s, cfs = np.array(x0s), np.array(cfs)
    errs_cf = recon_error_np(weights, cfs)

    m = cf_metrics(x0s, cfs, errs_cf, tau, trust_ref)
    print("\n=== Counterfactual quality metrics ===")
    print(f"  n            : {m['n']}")
    print(f"  Validity     : {m['validity']:.3f}   (fraction with error <= tau)")
    print(f"  Proximity    : {m['proximity']:.3f}   (mean L1 distance, scaled; lower better)")
    print(f"  Sparsity     : {m['sparsity']:.2f}    (mean # features changed; lower better)")
    print(f"  Plausibility : {m['plausibility']:.3f} (mean 1-NN distance to trustworthy; lower better)")

    valid = errs_cf <= tau
    if valid.any():
        k = np.argmin((np.abs(cfs[valid] - x0s[valid]) > SPARSITY_TOL).sum(axis=1))
        x0_o = x0s[valid][k] * rng_raw + lo_raw
        cf_o = cfs[valid][k] * rng_raw + lo_raw
        print("\n=== Case study: minimal change to restore trust ===")
        for c, a, bb in zip(cols, x0_o, cf_o):
            if abs(bb - a) > 1e-9:
                print(f"  {c:18s}: {a:14.4f}  ->  {bb:14.4f}   (delta {bb-a:+.4f})")

    with open(f"{OUTDIR}/cfg_metrics.json", "w") as f:
        json.dump({"metrics": m, "tau": tau, "n_explained": int(len(idx)),
                   "weights": [P_RECON, Q_FLIP, R_CLOSE], "immutable": IMMUTABLE_FEATURES,
                   "iters": N_ITERS}, f, indent=2)
    print(f"\nSaved {OUTDIR}/cfg_metrics.json")


if __name__ == "__main__":
    main()
