import shap
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load your data into X (features) and y (labels)
# X = ...
# y = ...

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# Normalize features
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# Define autoencoder
inp = layers.Input(shape=(X_train.shape[1],))
x = layers.Dense(64, activation='relu')(inp)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
out = layers.Dense(X_train.shape[1])(x)

autoencoder = tf.keras.Model(inp, out)
autoencoder.compile(optimizer='adam', loss='mse')

# Train autoencoder
autoencoder.fit(
    X_train, X_train,
    epochs=100, batch_size=8192,
    validation_split=0.3, verbose=1
)

# Encode data
encoded_train = autoencoder.predict(X_train)
encoded_test = autoencoder.predict(X_test)

# Train classifier
clf = LogisticRegression(max_iter=500)
clf.fit(encoded_train, y_train)
y_pred = clf.predict(encoded_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
DR = tp / (tp + fn)
FAR = fp / (fp + tn)
print(f"Detection Rate (DR): {DR:.4f}")
print(f"False Alarm Rate (FAR): {FAR:.4f}")

explainer = shap.KernelExplainer(autoencoder.predict, X_test.iloc[:100, :])
shap_values = explainer.shap_values(X_test.iloc[:100, :])
shap.summary_plot(shap_values, X_test.iloc[:100, :], feature_names=X.columns)
