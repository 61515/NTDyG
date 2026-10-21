#!/usr/bin/env python3
"""Recompute binary ranking and alert metrics from saved scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("Labels and scores must be nonempty vectors of equal length")
    if not set(np.unique(labels)).issubset({0, 1}) or not np.isfinite(scores).all():
        raise ValueError("Expected binary labels and finite anomaly scores")
    if len(np.unique(labels)) != 2:
        raise ValueError("Both classes are required for ROC-AUC")
    pred = scores >= threshold
    tp = int(((labels == 1) & pred).sum())
    fp = int(((labels == 0) & pred).sum())
    fn = int(((labels == 1) & ~pred).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"events": len(labels), "anomalies": int(labels.sum()),
            "anomaly_prevalence": float(labels.mean()),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "threshold": float(threshold), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "false_positives": fp}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--label-column", default="label")
    p.add_argument("--score-column", default="anomaly_score_class1_logit")
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--threshold", type=float,
                           help="A threshold already selected on validation data")
    selection.add_argument("--run-results", type=Path,
                           help="Read the frozen validation threshold from the run's results.json")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.output and a.output.exists():
        p.error("Output exists; select a new filename")
    if a.scores.suffix == ".npz":
        with np.load(a.scores, allow_pickle=False) as saved:
            labels, scores = saved["labels"], saved["scores"]
    else:
        frame = pd.read_csv(a.scores)
        labels, scores = frame[a.label_column].to_numpy(), frame[a.score_column].to_numpy()
    threshold = (json.loads(a.run_results.read_text())["validation_threshold_selected_by_f1"]
                 if a.run_results else a.threshold)
    report = metrics(labels, scores, threshold)
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
