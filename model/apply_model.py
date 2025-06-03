import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load input data
input_path = "output/labeled_flows.csv"
df = pd.read_csv(input_path)

# Drop rows with missing or infinite values
X_raw = df.drop(columns=["Type"], errors="ignore")
X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
X_raw.dropna(inplace=True)

# Load model parameters
param_dir = "model/params"
mean = np.loadtxt(os.path.join(param_dir, "mean.txt"), delimiter=",")
std = np.loadtxt(os.path.join(param_dir, "std.txt"), delimiter=",")
weights = np.loadtxt(os.path.join(param_dir, "weights.txt"), delimiter=",")
intercept = np.loadtxt(os.path.join(param_dir, "intercept.txt"), delimiter=",")

# Scale features
scaler = StandardScaler()
scaler.mean_ = mean
scaler.scale_ = std
X_scaled = scaler.transform(X_raw)

# Load model and predict
model = LogisticRegression()
model.coef_ = np.array([weights])
model.intercept_ = np.array(intercept)
model.classes_ = np.array([0, 1])

preds = model.predict(X_scaled)
labels = np.where(preds == 1, "Malicious", "Benign")

# Save predictions
df = df.iloc[:len(labels)].copy()
df["Predicted Label"] = labels
df.to_csv("output/predictions.csv", index=False)
print("Predictions saved to output/predictions.csv")