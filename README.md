# FlowClassifier

**FlowClassifier** is a lightweight machine learning pipeline that classifies network traffic flows as **benign** or **malicious** based on statistical flow features. This project is inspired by FlowMeter and builds on its flow export capabilities by adding an automated labeling and classification layer using logistic regression.

---

## Overview

- Processes flow-level CSVs derived from `.pcap` files
- Automatically labels flows based on known benign/malicious datasets
- Trains a logistic regression model for binary classification
- Applies the trained model to predict labels on new network flow data

---

## Setup

Install Python dependencies (PowerShell):

```powershell
pip install -r requirements.txt
```

Running the pipeline (PowerShell examples)

1) Run the full pipeline (labels -> merge -> train -> apply):

```powershell
python run_pipeline.py
```

2) Or run steps individually:

```powershell
# create labels (will look for files in data/ or data/flow_csv)
python scripts/label_flows.py

# merge labeled files into output/labeled_flows.csv
python scripts/merge_flows.py

# train model (reads output/labeled_flows.csv and saves params to model/params and model/model.joblib)
python model/train_model.py

# apply model (reads model/model.joblib or model/params and writes output/predictions.csv)
python model/apply_model.py

# evaluate predictions and run tuning/visualization (single combined script)
python scripts/evaluate_and_tune.py
```

Where to look for inputs/outputs

- Input CSVs expected in `data/` or `data/flow_csv` (the repo contains sample files under `data/`)
- Labeled/merged files are written to `output/`
- Model parameter files are saved to `model/params/`
- Predictions are written to `output/predictions.csv`

Evaluation and tuning

- The repo now contains a combined evaluation and tuning script: `scripts/evaluate_and_tune.py`.
- Run it to compute classification reports (predictions vs ground-truth), try several classifiers (logistic regression with multiple class weights and a RandomForest), compute precision/recall at multiple probability thresholds, and save ROC / Precision-Recall plots.
- The full text summary is written to `output/evaluation_and_tuning_report.txt` (this is the file we use to track the evaluation results).

---

## Notes and edge cases

- The training script expects `output/labeled_flows.csv` to be a clean numeric feature matrix with a `Type` column containing `Benign`/`Malicious`.
- If your CSV contains non-numeric columns (timestamps, IP addresses), drop or encode them before training.
- The `apply_model.py` script recreates a LogisticRegression object and sets its coefficients/intercept directly; this is a lightweight way to persist parameters but not the full model object. For production use, prefer `joblib` or `pickle` to save the estimator.

Random Forest — short technical explanation

- The tuning script includes a `RandomForestClassifier` as a candidate model. A Random Forest is an ensemble of decision trees:
	- Each tree is trained on a bootstrap sample (random subset with replacement) of the training data.
	- At each split in the tree, a random subset of features is considered (this decorrelates trees).
	- Predictions are made by averaging the probabilities (or majority voting) from individual trees, which reduces variance and often gives strong performance on structured/tabular data like flow statistics.

How the ML model classifies flow data (high level)

- Input features: each flow is represented by numeric statistics (for example: packet counts, byte counts, durations, average packet sizes, various timing statistics). The dataset used to produce the example output is stored in `output/labeled_flows.csv` and the evaluation summary is in `output/evaluation_and_tuning_report.txt`.
- Preprocessing: we select numeric columns only; missing or infinite values are dropped for simplicity. During training we fit a StandardScaler (for logistic models) so values are mean-centered and unit-variance — trees don't need scaling but linear models do.
- Model learning:
	- Logistic Regression learns a linear decision boundary in feature space; it uses regularization and class-weighting to compensate for class imbalance.
	- Random Forest builds many decision trees; each tree learns series of thresholds on feature values (for example, "if packet_count > 10 and duration < 2s then...") and the forest aggregates these to produce robust predictions.
- Why it works on flow statistics: malicious flows often differ from benign flows in patterns: sizes, timing, number of packets, directionality, and other statistical signatures. Models learn these correlations between features and the ground-truth label.
