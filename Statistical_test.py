import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.stats import friedmanchisquare, wilcoxon

# --- 0) Load data ---
df = pd.read_csv('dataset.csv')
y = df['label']

# Define feature sets (anonymized)
features_tms1 = [...]  # Full feature set
features_tms2 = [...]  # Subset 1
features_tms3 = [...]  # Subset 2 (e.g., SHAP-selected)

feature_sets = {
    'TMS1': features_tms1,
    'TMS2': features_tms2,
    'TMS3': features_tms3
}

# Cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Storage for fold-wise F1 scores
results = {name: [] for name in feature_sets}

for name, feat in feature_sets.items():
    X = df[feat]
    X = (X - X.mean()) / X.std().replace(0, 1)
    X_np = X.values
    y_np = y.values

    for train_idx, test_idx in kf.split(X_np, y_np):
        X_train, X_test = X_np[train_idx], X_np[test_idx]
        y_train, y_test = y_np[train_idx], y_np[test_idx]

        # Autoencoder
        inp = layers.Input(shape=(X_train.shape[1],))
        x = layers.Dense(64, activation='relu')(inp)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        out = layers.Dense(X_train.shape[1], activation='linear')(x)
        ae = models.Model(inp, out)
        ae.compile(optimizer='adam', loss='mse')
        ae.fit(X_train, X_train, epochs=100, batch_size=8192, verbose=0)

        # Encode
        Z_train = ae.predict(X_train, verbose=0)
        Z_test = ae.predict(X_test, verbose=0)

        # Classifier
        clf = LogisticRegression(max_iter=500)
        clf.fit(Z_train, y_train)
        y_pred = clf.predict(Z_test)

        # Metrics
        f1 = f1_score(y_test, y_pred)
        results[name].append(f1)

# Convert results to arrays
tms1_scores = np.array(results['TMS1'])
tms2_scores = np.array(results['TMS2'])
tms3_scores = np.array(results['TMS3'])

# Friedman test
stat_f, p_f = friedmanchisquare(tms1_scores, tms2_scores, tms3_scores)

# Wilcoxon pairwise tests
stat_w1, p_w1 = wilcoxon(tms1_scores, tms2_scores)
stat_w2, p_w2 = wilcoxon(tms1_scores, tms3_scores)
stat_w3, p_w3 = wilcoxon(tms2_scores, tms3_scores)

# Report
print("F1-scores (5-fold CV):")
for name in results:
    arr = np.array(results[name])
    print(f"{name}: {arr.mean():.3f} ± {arr.std():.3f}")

print(f"\nFriedman test: statistic={stat_f:.3f}, p-value={p_f:.4f}")
print("Wilcoxon tests (TMS1 vs TMS2, TMS1 vs TMS3, TMS2 vs TMS3):")
print(f"TMS1 vs TMS2: W={stat_w1:.3f}, p={p_w1:.4f}")
print(f"TMS1 vs TMS3: W={stat_w2:.3f}, p={p_w2:.4f}")
print(f"TMS2 vs TMS3: W={stat_w3:.3f}, p={p_w3:.4f}")
