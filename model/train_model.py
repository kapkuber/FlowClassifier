import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, f1_score

# Load and prepare data
data_path = "output/labeled_flows.csv"
df = pd.read_csv(data_path)

# Drop missing or infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Encode labels
df["Label"] = df["Type"].apply(lambda x: 1 if x == "Malicious" else 0)
X = df.drop(columns=["Type", "Label"], errors="ignore")
y = df["Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression + GridSearch
param_grid = {
    "class_weight": [
        {0: 0.25, 1: 99.75},
        {0: 0.50, 1: 99.50},
        {0: 1.00, 1: 99.00}
    ],
    "penalty": ["l1", "l2"],
    "C": np.arange(0.1, 1.0, 0.2),
    "fit_intercept": [True, False]
}

lr = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
grid = GridSearchCV(lr, param_grid, scoring="roc_auc", cv=5, n_jobs=-1)
grid.fit(X_train_scaled, y_train)

# Evaluate
y_pred = grid.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model parameters
model = grid.best_estimator_
param_dir = "model/params"
os.makedirs(param_dir, exist_ok=True)

np.savetxt(os.path.join(param_dir, "weights.txt"), model.coef_[0], delimiter=",")
np.savetxt(os.path.join(param_dir, "intercept.txt"), model.intercept_, delimiter=",")
np.savetxt(os.path.join(param_dir, "mean.txt"), scaler.mean_, delimiter=",")
np.savetxt(os.path.join(param_dir, "std.txt"), scaler.scale_, delimiter=",")

print("Model training complete. Parameters saved to model/params/")