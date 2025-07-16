import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

# Load dataset
df = pd.read_csv('dataset.csv')
X = df.drop('label', axis=1)
y = df['label']

# Split into train/val/test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_train = X_trainval[y_trainval == 0].copy()
X_val = X_train.sample(frac=0.2, random_state=1)
X_train = X_train.drop(X_val.index)

# Normalize
mean = X_train.mean()
std = X_train.std().replace(0, 1)
X_train_s = (X_train - mean) / std
X_val_s = (X_val - mean) / std
X_test_s = (X_test - mean) / std

# Build and train Autoencoder
inp = layers.Input(shape=(X_train_s.shape[1],))
x = layers.Dense(64, activation='relu')(inp)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
out = layers.Dense(X_train_s.shape[1], activation='linear')(x)

autoencoder = models.Model(inp, out)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(
    X_train_s, X_train_s,
    epochs=100, batch_size=8192,
    validation_data=(X_val_s, X_val_s),
    verbose=1
)

# Compute reconstruction errors
def recon_error(model, X):
    recon = model.predict(X)
    return np.mean((X - recon) ** 2, axis=1)

errs_val = recon_error(autoencoder, X_val_s)
errs_test = recon_error(autoencoder, X_test_s)

# Threshold via percentile rule
threshold = np.percentile(errs_val, 99)

# Predict
y_pred = (errs_test > threshold).astype(int)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
DR = tp / (tp + fn)
FAR = fp / (fp + tn)
print(f"Detection Rate (DR): {DR:.4f}")
print(f"False Alarm Rate (FAR): {FAR:.4f}")
