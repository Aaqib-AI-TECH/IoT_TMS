import os
import time
import json
import random
import platform
import statistics
import numpy as np
import pandas as pd

# ----------------------------- configuration -----------------------------
DATA_PATH = "dataset.csv"          # same CSV used for Config A
LABEL_COL = "label"
TRUSTWORTHY_LABEL = 0
HIDDEN = (64, 32, 64)
N_WARMUP = 50
N_REPEAT = 1000                    # per-sample timing repeats
BATCH_FOR_THROUGHPUT = None        # None -> use full test set
RUN_CONFIG_B = True                # also time AE-features + logistic regression
RUN_SHAP = True                    # time one-sample SHAP explanation (slow; set False to skip)
SHAP_BACKGROUND = 100
SEED = 42
OUTDIR = "outputs_benchmark"

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)


# ----------------------------- environment -----------------------------
def hardware_info():
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import psutil
        info["logical_cpus"] = psutil.cpu_count(logical=True)
        info["physical_cpus"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        info["logical_cpus"] = os.cpu_count()
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
        info["tf_gpus"] = len(tf.config.list_physical_devices("GPU"))
    except Exception:
        info["tensorflow"] = "not available"
    return info


# ----------------------------- model + data -----------------------------
def load_features():
    df = pd.read_csv(DATA_PATH)
    cols = SELECTED_FEATURES if SELECTED_FEATURES else [c for c in df.columns if c != LABEL_COL]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    X = df[cols].astype(float)
    # leakage-free standardisation on trustworthy rows (timing is unaffected, but keep it faithful)
    y = df[LABEL_COL].astype(int)
    Xt = X[y == TRUSTWORTHY_LABEL]
    mean, std = Xt.mean(), Xt.std().replace(0, 1)
    Xs = ((X - mean) / std).values.astype(np.float32)
    return Xs, y.values


def build_model(n_features, seed=SEED):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    tf.random.set_seed(seed)
    inp = layers.Input(shape=(n_features,))
    x = layers.Dense(HIDDEN[0], activation="relu")(inp)
    x = layers.Dense(HIDDEN[1], activation="relu")(x)
    x = layers.Dense(HIDDEN[2], activation="relu")(x)
    out = layers.Dense(n_features, activation="linear")(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="mse")
    return m


def extract_weights(model):
    """Return list of (W, b) for the Dense layers, for a pure-NumPy forward pass."""
    pairs = []
    for layer in model.layers:
        w = layer.get_weights()
        if len(w) == 2:
            pairs.append((w[0].astype(np.float32), w[1].astype(np.float32)))
    return pairs


def numpy_forward(weights, x):
    """Deployable AE forward pass in NumPy. ReLU on hidden layers, linear output."""
    h = x
    for i, (W, b) in enumerate(weights):
        h = h @ W + b
        if i < len(weights) - 1:
            h = np.maximum(h, 0.0)   # ReLU
    return h


# ----------------------------- timing -----------------------------
def summarize(times_s):
    a = np.array(times_s, dtype=float) * 1e3  # -> milliseconds
    return {
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "p99_ms": float(np.percentile(a, 99)),
        "std_ms": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "n": int(len(a)),
    }


def time_per_call(fn, n_warmup=N_WARMUP, n_repeat=N_REPEAT):
    for _ in range(n_warmup):
        fn()
    out = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return summarize(out)


# ----------------------------- main -----------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    env = hardware_info()
    print("=== Environment ===")
    for k, v in env.items():
        print(f"  {k}: {v}")

    Xs, y = load_features()
    n_features = Xs.shape[1]
    x1 = Xs[:1]                       # a single sample
    results = {"environment": env, "n_features": n_features, "n_test": int(len(Xs))}

    model = build_model(n_features)
    weights = extract_weights(model)

    # ---- memory footprint ----
    n_params = int(model.count_params())
    model_kb = n_params * 4 / 1024.0
    results["model"] = {"parameters": n_params, "footprint_KB_float32": round(model_kb, 2)}
    print(f"\n=== Model footprint ===\n  parameters: {n_params}  "
          f"(~{model_kb:.1f} KB at float32)")

    # ---- detection latency (a) Keras eager single sample ----
    import tensorflow as tf
    xt = tf.convert_to_tensor(x1)
    keras_single = time_per_call(lambda: model(xt, training=False))

    # ---- detection latency (b) NumPy forward, single sample ----
    def numpy_detect():
        recon = numpy_forward(weights, x1)
        err = float(np.mean((x1 - recon) ** 2))
        return err > 0.0          # threshold compare (tau constant; timing identical)
    numpy_single = time_per_call(numpy_detect)

    # ---- detection latency (c) batched amortised throughput ----
    batch = Xs if BATCH_FOR_THROUGHPUT is None else Xs[:BATCH_FOR_THROUGHPUT]
    for _ in range(3):
        model.predict(batch, verbose=0)         # warm up
    t0 = time.perf_counter()
    model.predict(batch, verbose=0)
    batch_s = time.perf_counter() - t0
    amortised_ms = batch_s / len(batch) * 1e3
    throughput = len(batch) / batch_s

    results["detection_latency"] = {
        "keras_eager_single_sample": keras_single,
        "numpy_forward_single_sample": numpy_single,
        "batched_amortised_ms_per_sample": round(amortised_ms, 5),
        "throughput_samples_per_sec": round(throughput, 1),
    }
    print("\n=== Detection latency (Config A) ===")
    print(f"  Keras eager, 1 sample : mean {keras_single['mean_ms']:.3f} ms "
          f"(p99 {keras_single['p99_ms']:.3f})")
    print(f"  NumPy forward, 1 sample: mean {numpy_single['mean_ms']:.5f} ms "
          f"(p99 {numpy_single['p99_ms']:.5f})")
    print(f"  Batched amortised      : {amortised_ms:.5f} ms/sample "
          f"({throughput:,.0f} samples/sec)")

    # ---- optional Config B: + logistic regression ----
    if RUN_CONFIG_B:
        from sklearn.linear_model import LogisticRegression
        feats = model.predict(Xs, verbose=0)
        clf = LogisticRegression(max_iter=200).fit(feats, y)
        f1 = feats[:1]
        clf_single = time_per_call(lambda: clf.predict(f1))
        results["config_b_classifier_single_sample"] = clf_single
        print(f"\n=== Config B extra (LR predict, 1 sample) ===\n"
              f"  mean {clf_single['mean_ms']:.5f} ms")

    # ---- explanation latency: SHAP (separate, slow) ----
    if RUN_SHAP:
        try:
            import shap
            bg = Xs[np.random.choice(len(Xs), size=min(SHAP_BACKGROUND, len(Xs)), replace=False)]
            explainer = shap.KernelExplainer(lambda d: model.predict(d, verbose=0), bg)
            t0 = time.perf_counter()
            explainer.shap_values(x1, silent=True)
            shap_s = time.perf_counter() - t0
            results["explanation_latency_shap_single_sample_ms"] = round(shap_s * 1e3, 1)
            print(f"\n=== Explanation latency (SHAP, 1 sample, bg={SHAP_BACKGROUND}) ===\n"
                  f"  {shap_s*1e3:,.0f} ms  (reported separately from detection)")
        except Exception as e:
            print(f"\n[SHAP skipped: {e}]")

    # ---- peak memory during a batch (Python-level + process RSS) ----
    import tracemalloc
    tracemalloc.start()
    model.predict(batch, verbose=0)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem = {"python_peak_MB_during_batch": round(peak / 1e6, 3)}
    try:
        import psutil
        mem["process_rss_MB"] = round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:
        pass
    results["memory"] = mem
    print(f"\n=== Memory ===\n  model footprint: {model_kb:.1f} KB | "
          f"python peak (batch): {mem['python_peak_MB_during_batch']} MB | "
          f"process RSS: {mem.get('process_rss_MB', 'n/a')} MB")

    with open(f"{OUTDIR}/benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUTDIR}/benchmark.json")


if __name__ == "__main__":
    main()
