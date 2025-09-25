import os
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from joblib import dump

# Load and prepare data
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="output/labeled_flows.csv", help="labeled flows CSV")
parser.add_argument("--params-dir", default="model/params", help="directory to save model parameters")
parser.add_argument("--model-file", default="model/model.joblib", help="joblib file to save full pipeline")
args = parser.parse_args()

data_path = args.input
df = pd.read_csv(data_path)

# Drop missing or infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Encode labels
df["Label"] = df["Type"].apply(lambda x: 1 if x == "Malicious" else 0)
X = df.drop(columns=["Type", "Label"], errors="ignore")

# Select numeric columns only (drop IPs, strings, etc.)
X = X.select_dtypes(include=[np.number])
if X.shape[1] == 0:
    raise ValueError("No numeric feature columns found in input CSV. Please provide numeric features or preprocess the CSV to remove non-numeric columns.")

y = df["Label"]

# align y with X (drop rows where X had NaNs removed earlier)
valid_idx = X.index
X = X.loc[valid_idx]
y = y.loc[valid_idx]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features and create pipeline
scaler = StandardScaler()

# Logistic Regression + GridSearch wrapped in a pipeline
pipeline = Pipeline([('scaler', scaler), ('clf', LogisticRegression(random_state=42, solver='liblinear', max_iter=1000))])

param_grid = {
    'clf__class_weight': [
        {0: 0.25, 1: 99.75},
        {0: 0.50, 1: 99.50},
        {0: 1.00, 1: 99.00}
    ],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': np.arange(0.1, 1.0, 0.2),
    'clf__fit_intercept': [True, False]
}

grid = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

# Evaluate
y_pred = grid.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
try:
    auc = roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])
except Exception:
    auc = float('nan')
print("AUC:", auc)
print("F1:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model parameters and full pipeline
best = grid.best_estimator_
param_dir = args.params_dir
os.makedirs(param_dir, exist_ok=True)

# If the pipeline contains a fitted scaler and classifier, extract params
try:
    clf = best.named_steps['clf']
    sc = best.named_steps['scaler']
    weights = clf.coef_[0]
    intercept = clf.intercept_
    mean = sc.mean_
    std = sc.scale_
    np.savetxt(os.path.join(param_dir, "weights.txt"), weights, delimiter=",")
    np.savetxt(os.path.join(param_dir, "intercept.txt"), intercept, delimiter=",")
    np.savetxt(os.path.join(param_dir, "mean.txt"), mean, delimiter=",")
    np.savetxt(os.path.join(param_dir, "std.txt"), std, delimiter=",")
except Exception:
    # fallback: try to save from underlying objects if schema differs
    pass

# Save feature names so apply_model can select the same columns
feat_file = os.path.join(param_dir, 'feature_names.txt')
with open(feat_file, 'w', encoding='utf-8') as f:
    for c in X.columns:
        f.write(f"{c}\n")

# Save a pipeline (scaler + classifier) using joblib
model_file = args.model_file
os.makedirs(os.path.dirname(model_file), exist_ok=True)
# create a pipeline with the fitted scaler and classifier for easy load
try:
    # best is a fitted Pipeline
    dump(best, model_file)
    print(f"Saved joblib pipeline to {model_file}")
except Exception as e:
    print("Warning: could not save joblib pipeline:", e)

print("Model training complete. Parameters saved to", param_dir)