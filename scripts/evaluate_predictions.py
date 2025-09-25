#!/usr/bin/env python3
"""Evaluate model predictions against ground-truth labels.

Usage:
  python scripts/evaluate_predictions.py 
  python scripts/evaluate_predictions.py --pred output/predictions.csv --gt output/labeled_flows.csv

The script will print a sklearn classification_report and confusion matrix.
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_data(pred_path, gt_path=None):
    preds = pd.read_csv(pred_path)

    if gt_path is not None and gt_path != '':
        gt = pd.read_csv(gt_path)
    else:
        # try to use Type column if present in preds
        gt = None

    return preds, gt


def extract_labels(preds_df, gt_df=None):
    # Predicted labels: try multiple column names
    pred_col_candidates = ['Predicted Label', 'Predicted_Label', 'predicted_label', 'prediction', 'label']
    pred_col = None
    for c in pred_col_candidates:
        if c in preds_df.columns:
            pred_col = c
            break

    if pred_col is None:
        # try numeric 0/1 in a single column
        # if only one extra column beyond features is present, try that
        possible = [c for c in preds_df.columns if c not in ('Type',)]
        if len(possible) == 1:
            pred_col = possible[0]
        else:
            raise ValueError("Could not locate predicted label column in predictions CSV. Expected 'Predicted Label'.")

    y_pred = preds_df[pred_col].values

    # normalize predicted values to strings 'Benign'/'Malicious'
    if np.issubdtype(y_pred.dtype, np.number):
        y_pred = np.where(y_pred == 1, 'Malicious', 'Benign')
    else:
        y_pred = y_pred.astype(str)

    # ground truth labels
    if 'Type' in preds_df.columns and gt_df is None:
        y_true = preds_df['Type'].values.astype(str)
    elif gt_df is not None:
        if 'Type' in gt_df.columns:
            # align by row order: use min length
            n = min(len(gt_df), len(preds_df))
            y_true = gt_df['Type'].values[:n].astype(str)
            y_pred = y_pred[:n]
        else:
            raise ValueError("Ground truth file does not contain a 'Type' column.")
    else:
        raise ValueError("No ground truth available (no 'Type' column found and no --gt provided)")

    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', default='output/predictions.csv', help='Predictions CSV path')
    parser.add_argument('--gt', default='output/labeled_flows.csv', help='Ground-truth labeled CSV path (must contain Type column)')
    parser.add_argument('--labels', nargs='+', default=['Benign', 'Malicious'], help='Label order for reports')
    args = parser.parse_args()

    preds_df, gt_df = load_data(args.pred, args.gt)

    y_true, y_pred = extract_labels(preds_df, gt_df)

    print('\nEvaluation summary')
    print('------------------')
    print('Samples evaluated:', len(y_true))
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('\nClassification report:\n')
    print(classification_report(y_true, y_pred, labels=args.labels, zero_division=0))
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred, labels=args.labels))


if __name__ == '__main__':
    main()
