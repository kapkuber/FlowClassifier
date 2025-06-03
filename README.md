# FlowClassifier

**FlowClassifier** is a lightweight machine learning pipeline that classifies network traffic flows as **benign** or **malicious** based on statistical flow features. This project is inspired by [FlowMeter](https://github.com/deepfence/FlowMeter) by Deepfence and builds on its flow export capabilities by adding an automated labeling and classification layer using logistic regression.

---

## 🔍 Overview

- Processes flow-level CSVs derived from `.pcap` files
- Automatically labels flows based on known benign/malicious datasets
- Trains a logistic regression model for binary classification
- Applies the trained model to predict labels on new network flow data

---

## ⚙️ Setup

Install Python dependencies:
```bash
pip install pandas scikit-learn numpy
