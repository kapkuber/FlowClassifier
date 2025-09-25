#!/usr/bin/env python3
"""Combined evaluation and tuning script.

It runs two main actions:
 1) Evaluate existing predictions in --pred against ground-truth in --gt (classification report + confusion matrix)
 2) Run classifier comparisons, compute precision/recall at thresholds, and save ROC/PR plots

Outputs:
 - prints evaluation and tuning summaries to stdout
 - saves plots to --outdir (default: output/)
 - writes a text summary to <outdir>/evaluation_and_tuning_report.txt

Usage:
  python scripts/evaluate_and_tune.py
  python scripts/evaluate_and_tune.py --pred output/predictions.csv --gt output/labeled_flows.csv --outdir output
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_predictions(pred_path):
    return pd.read_csv(pred_path)


def load_ground_truth(gt_path):
    return pd.read_csv(gt_path)


def extract_true_pred(preds_df, gt_df=None):
    # Detect predicted label column
    pred_col_candidates = ['Predicted Label', 'Predicted_Label', 'predicted_label', 'prediction', 'label']
    pred_col = next((c for c in pred_col_candidates if c in preds_df.columns), None)
    if pred_col is None:
        # fallback: take last column
        pred_col = preds_df.columns[-1]

    y_pred = preds_df[pred_col].values
    if np.issubdtype(y_pred.dtype, np.number):
        y_pred = np.where(y_pred == 1, 'Malicious', 'Benign')
    else:
        y_pred = y_pred.astype(str)

    if gt_df is not None:
        if 'Type' not in gt_df.columns:
            raise ValueError("Ground truth file must contain a 'Type' column")
        n = min(len(gt_df), len(y_pred))
        y_true = gt_df['Type'].values[:n].astype(str)
        y_pred = y_pred[:n]
    else:
        if 'Type' in preds_df.columns:
            y_true = preds_df['Type'].values.astype(str)
            # align lengths
            n = min(len(y_true), len(y_pred))
            y_true = y_true[:n]
            y_pred = y_pred[:n]
        else:
            raise ValueError('No ground-truth Type column found in predictions and no gt file provided')

    return y_true, y_pred


def print_evaluation(y_true, y_pred, out_lines):
    acc = accuracy_score(y_true, y_pred)
    out_lines.append(f"Samples evaluated: {len(y_true)}")
    out_lines.append(f"Accuracy: {acc:.6f}")
    out_lines.append('\nClassification report:\n')
    report = classification_report(y_true, y_pred, zero_division=0)
    out_lines.append(report)
    out_lines.append('Confusion matrix:')
    cm = confusion_matrix(y_true, y_pred)
    out_lines.append(str(cm))
    # also print to stdout
    print('\nEvaluation summary')
    print('------------------')
    print('\n'.join(out_lines[:3]))
    print(report)
    print('Confusion matrix:')
    print(cm)


def prepare_numeric_features(gt_path):
    df = pd.read_csv(gt_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    if 'Type' not in df.columns:
        raise ValueError("Ground truth file must contain a 'Type' column for tuning")
    df['Label'] = df['Type'].apply(lambda x: 1 if x == 'Malicious' else 0)
    X = df.drop(columns=['Type', 'Label'], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        raise ValueError('No numeric features found in ground truth CSV for tuning')
    y = df.loc[X.index, 'Label']
    return X, y


def try_classifiers(X_train, X_test, y_train, y_test):
    results = []
    classifiers = []
    for cw in [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 10}, {0: 1, 1: 50}]:
        classifiers.append(('LogReg_cw_' + str(list(cw.values())), LogisticRegression(class_weight=cw, solver='liblinear', max_iter=1000)))
    classifiers.append(('RandomForest_default', RandomForestClassifier(n_estimators=100, random_state=42)))

    for name, clf in classifiers:
        if isinstance(clf, LogisticRegression):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_train)
            Xte = scaler.transform(X_test)
        else:
            Xtr = X_train.values
            Xte = X_test.values

        clf.fit(Xtr, y_train)

        if hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba(Xte)[:, 1]
        else:
            try:
                probs = clf.decision_function(Xte)
            except Exception:
                probs = clf.predict(Xte)

        preds = (probs >= 0.5).astype(int)
        pr = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

        results.append({'name': name, 'precision': pr, 'recall': rec, 'f1': f1, 'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr, 'probs': probs})

    return results


def precision_at_thresholds(y_true, probs, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)
    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        pr = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        rows.append((t, pr, rec))
    return rows


def plot_and_save(results, best, y_test, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for r in results:
        plt.plot(r['fpr'], r['tpr'], label=f"{r['name']} (AUC={r['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(outdir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, best['probs'])
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"{best['name']} (AUC={pr_auc:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    pr_path = os.path.join(outdir, 'pr_curve.png')
    plt.savefig(pr_path)
    plt.close()

    return roc_path, pr_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', default='output/predictions.csv', help='predictions CSV')
    parser.add_argument('--gt', default='output/labeled_flows.csv', help='ground-truth labeled CSV')
    parser.add_argument('--outdir', default='output', help='output directory for plots and report')
    args = parser.parse_args()

    out_lines = []
    preds_df = load_predictions(args.pred)
    gt_df = load_ground_truth(args.gt)
    y_true, y_pred = extract_true_pred(preds_df, gt_df)

    out_lines.append(f"Samples evaluated: {len(y_true)}")
    out_lines.append(f"Accuracy: {accuracy_score(y_true, y_pred):.6f}")
    report = classification_report(y_true, y_pred, zero_division=0)
    out_lines.append('\nClassification report:\n')
    out_lines.append(report)
    out_lines.append('Confusion matrix:')
    out_lines.append(str(confusion_matrix(y_true, y_pred)))

    # Tuning + visualization
    X, y = prepare_numeric_features(args.gt)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    results = try_classifiers(X_train, X_test, y_train, y_test)

    # add tuning summary to out_lines and print
    out_lines.append('\nClassifier summary:')
    for r in results:
        out_lines.append(f"{r['name']}: precision={r['precision']:.3f}, recall={r['recall']:.3f}, f1={r['f1']:.3f}, roc_auc={r['roc_auc']:.3f}")
        print(f"{r['name']}: precision={r['precision']:.3f}, recall={r['recall']:.3f}, f1={r['f1']:.3f}, roc_auc={r['roc_auc']:.3f}")

    best = max(results, key=lambda x: x['roc_auc'])
    out_lines.append(f"\nBest classifier by ROC AUC: {best['name']} (AUC={best['roc_auc']:.3f})")

    thr_rows = precision_at_thresholds(y_test, best['probs'])
    out_lines.append('\nPrecision/Recall at thresholds for best classifier:')
    for t, pr, rec in thr_rows:
        out_lines.append(f" threshold={t:.2f}: precision={pr:.3f}, recall={rec:.3f}")

    roc_path, pr_path = plot_and_save(results, best, y_test, args.outdir)
    out_lines.append(f"Saved ROC to {roc_path}")
    out_lines.append(f"Saved PR to {pr_path}")

    # write a text report
    report_path = os.path.join(args.outdir, 'evaluation_and_tuning_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))

    print('\nEvaluation + tuning finished. Report saved to', report_path)


if __name__ == '__main__':
    main()
