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

# train model (reads output/labeled_flows.csv and saves params to model/params)
python model/train_model.py

# apply model (reads model/params and writes output/predictions.csv)
python model/apply_model.py
```

Where to look for inputs/outputs

- Input CSVs expected in `data/` or `data/flow_csv` (the repo contains sample files under `data/`)
- Labeled/merged files are written to `output/`
- Model parameter files are saved to `model/params/`
- Predictions are written to `output/predictions.csv`

---

## Notes and edge cases

- The training script expects `output/labeled_flows.csv` to be a clean numeric feature matrix with a `Type` column containing `Benign`/`Malicious`.
- If your CSV contains non-numeric columns (timestamps, IP addresses), drop or encode them before training.
- The `apply_model.py` script recreates a LogisticRegression object and sets its coefficients/intercept directly; this is a lightweight way to persist parameters but not the full model object. For production use, prefer `joblib` or `pickle` to save the estimator.

If you want, I can run a small smoke test in this workspace (syntax run) or help you adapt the pipeline to accept raw pcap files.
