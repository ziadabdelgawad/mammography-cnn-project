from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import ARTIFACTS_DIR, MODELS_DIR
from .data import load_datasets


def labels_and_probs(model, ds):
    ys, ps = [], []
    for x, y in ds:
        p = model.predict(x, verbose=0).reshape(-1)
        ys.extend(y.numpy().reshape(-1).astype(int).tolist())
        ps.extend(p.tolist())
    return np.asarray(ys), np.asarray(ps)


def threshold_from_validation(y, p):
    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.isfinite(thresholds)
    scores = np.where(finite, tpr - fpr, -np.inf)
    idx = int(np.argmax(scores))
    return float(thresholds[idx])


def metric_dict(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "sensitivity_recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_confusion(y, p, threshold, out):
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], ["Benign", "Malignant"])
    ax.set_yticks([0, 1], ["Benign", "Malignant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_roc(y, p, out):
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_pr(y, p, out):
    precision, recall, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"PR-AUC = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def evaluate_one(name: str, datasets):
    model_path = MODELS_DIR / f"{name}.keras"
    if not model_path.is_file():
        print(f"SKIP {name}: {model_path} not found")
        return None
    print(f"\nEvaluating {name}...")
    model = tf.keras.models.load_model(model_path)
    y_val, p_val = labels_and_probs(model, datasets["validation"])
    threshold = threshold_from_validation(y_val, p_val)
    val_metrics = metric_dict(y_val, p_val, threshold)

    y_test, p_test = labels_and_probs(model, datasets["test"])
    test_metrics = metric_dict(y_test, p_test, threshold)

    out = ARTIFACTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "model": name,
        "threshold_selected_on_validation": threshold,
        "validation": val_metrics,
        "test": test_metrics,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    with (out / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true_label", "malignant_probability", "predicted_label"])
        for yy, pp in zip(y_test, p_test):
            w.writerow([int(yy), float(pp), int(pp >= threshold)])

    plot_confusion(y_test, p_test, threshold, out / "confusion_matrix.png")
    plot_roc(y_test, p_test, out / "roc_curve.png")
    plot_pr(y_test, p_test, out / "pr_curve.png")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "regularized", "transfer"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and not args.model:
        parser.error("Use --all or --model NAME")

    datasets = load_datasets()
    names = ["baseline", "regularized", "transfer"] if args.all else [args.model]
    results = [r for n in names if (r := evaluate_one(n, datasets)) is not None]
    if results:
        summary = ARTIFACTS_DIR / "model_comparison.csv"
        with summary.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "validation_roc_auc", "test_roc_auc", "test_accuracy", "test_sensitivity", "test_specificity", "test_f1"])
            for r in results:
                w.writerow([
                    r["model"], r["validation"]["roc_auc"], r["test"]["roc_auc"],
                    r["test"]["accuracy"], r["test"]["sensitivity_recall"],
                    r["test"]["specificity"], r["test"]["f1"],
                ])
        print(f"\nComparison saved to {summary}")


if __name__ == "__main__":
    main()
