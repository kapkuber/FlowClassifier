import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import load

# Load input data
input_path = "output/labeled_flows.csv"
df = pd.read_csv(input_path)

# Drop rows with missing or infinite values
X_raw = df.drop(columns=["Type"], errors="ignore")
X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
X_raw.dropna(inplace=True)

param_dir = "model/params"
joblib_path = "model/model.joblib"

if os.path.exists(joblib_path):
	# Load full pipeline
	try:
		pipeline = load(joblib_path)
		# try to select the features saved earlier if they exist
		feat_file = os.path.join(param_dir, 'feature_names.txt')
		if os.path.exists(feat_file):
			with open(feat_file, 'r', encoding='utf-8') as f:
				feats = [l.strip() for l in f if l.strip()]
			X = X_raw.loc[:, X_raw.columns.intersection(feats)]
			# Reorder to match saved feature order
			X = X.reindex(columns=feats)
		else:
			# use numeric columns only
			X = X_raw.select_dtypes(include=[np.number])

		preds = pipeline.predict(X)
		labels = np.where(preds == 1, "Malicious", "Benign")
	except Exception as e:
		raise RuntimeError(f"Failed to load joblib pipeline: {e}")
else:
	# Fallback: load param files and reconstruct classifier
	mean = np.loadtxt(os.path.join(param_dir, "mean.txt"), delimiter=",")
	std = np.loadtxt(os.path.join(param_dir, "std.txt"), delimiter=",")
	weights = np.loadtxt(os.path.join(param_dir, "weights.txt"), delimiter=",")
	intercept = np.loadtxt(os.path.join(param_dir, "intercept.txt"), delimiter=",")

	# Scale features
	scaler = StandardScaler()
	scaler.mean_ = mean
	scaler.scale_ = std
	try:
		scaler.n_features_in_ = len(mean)
	except Exception:
		pass

	# Select numeric columns to match saved feature count
	X = X_raw.select_dtypes(include=[np.number])
	X_scaled = scaler.transform(X)

	# Load model and predict
	model = LogisticRegression()
	model.coef_ = np.array([weights])
	model.intercept_ = np.array(intercept)
	model.classes_ = np.array([0, 1])
	try:
		model.n_features_in_ = X_scaled.shape[1]
	except Exception:
		pass

	preds = model.predict(X_scaled)
	labels = np.where(preds == 1, "Malicious", "Benign")

# Save predictions
df = df.iloc[:len(labels)].copy()
df["Predicted Label"] = labels
df.to_csv("output/predictions.csv", index=False)
print("Predictions saved to output/predictions.csv")