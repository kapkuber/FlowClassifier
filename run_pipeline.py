"""Small helper to run the full FlowClassifier pipeline.

Usage: python run_pipeline.py

It will run the following steps in order:
 - scripts/label_flows.py
 - scripts/merge_flows.py
 - model/train_model.py
 - model/apply_model.py

Outputs:
 - output/benign_labeled.csv, output/malicious_labeled.csv
 - output/labeled_flows.csv
 - model/params/* (weights, intercept, mean, std)
 - output/predictions.csv
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

def run(cmd):
    print(f"Running: {cmd}")
    subprocess.run([sys.executable] + cmd.split(), check=True)

def main():
    steps = [
        "scripts/label_flows.py",
        "scripts/merge_flows.py",
        "model/train_model.py",
        "model/apply_model.py",
    ]

    for s in steps:
        run(s)

    print("Pipeline finished. Check output/ and model/params/")

if __name__ == '__main__':
    main()
