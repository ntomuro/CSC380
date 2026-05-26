"""
Evaluation utilities for the prompt injection detection system.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

LABEL_ORDER = ["Benign", "Suspicious", "Injection"]


def compute_metrics(
    ground_truth: list[str],
    predictions: list[str],
) -> dict:
    """
    Compute classification metrics for the detection system.

    Returns a dict with:
        accuracy          : float
        macro_f1          : float
        weighted_f1       : float
        per_class_report  : dict  (sklearn classification_report dict)
        confusion_matrix  : list[list[int]]  (rows=true, cols=predicted)
        label_order       : list[str]
    """
    acc = accuracy_score(ground_truth, predictions)
    macro_f1 = f1_score(
        ground_truth, predictions, labels=LABEL_ORDER, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        ground_truth, predictions, labels=LABEL_ORDER, average="weighted", zero_division=0
    )
    report = classification_report(
        ground_truth, predictions, labels=LABEL_ORDER, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(ground_truth, predictions, labels=LABEL_ORDER).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_report": report,
        "confusion_matrix": cm,
        "label_order": LABEL_ORDER,
    }


def print_metrics(metrics: dict) -> None:
    """Print a compact summary of metrics to stdout."""
    print(f"Accuracy    : {metrics['accuracy']:.3f}")
    print(f"Macro F1    : {metrics['macro_f1']:.3f}")
    print(f"Weighted F1 : {metrics['weighted_f1']:.3f}")
    print()
    labels = metrics["label_order"]
    report = metrics["per_class_report"]
    header = f"{'Class':<16}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}  {'Support':>8}"
    print(header)
    print("-" * len(header))
    for label in labels:
        row = report[label]
        print(
            f"{label:<16}  {row['precision']:>9.3f}  {row['recall']:>9.3f}"
            f"  {row['f1-score']:>9.3f}  {int(row['support']):>8}"
        )


def plot_confusion_matrix(
    metrics: dict,
    title: str = "Confusion Matrix — Prompt Injection Detection",
) -> plt.Figure:
    """Return a seaborn heatmap figure of the confusion matrix."""
    cm = metrics["confusion_matrix"]
    labels = metrics["label_order"]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    return fig


def error_analysis(
    ground_truth: list[str],
    predictions: list[str],
    conversations: list[dict],
) -> dict:
    """
    Separate prediction errors by type.

    Returns:
        false_positives  : Benign conversations classified as Suspicious or Injection
        false_negatives  : Injection conversations classified as Benign
        suspicious_errors: Suspicious conversations classified as Benign or Injection
    """
    false_positives, false_negatives, suspicious_errors = [], [], []
    for gt, pred, conv in zip(ground_truth, predictions, conversations):
        if gt == "Benign" and pred != "Benign":
            false_positives.append({"conversation": conv, "predicted": pred})
        elif gt == "Injection" and pred == "Benign":
            false_negatives.append({"conversation": conv, "predicted": pred})
        elif gt == "Suspicious" and pred in ("Benign", "Injection"):
            suspicious_errors.append({"conversation": conv, "predicted": pred})
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "suspicious_errors": suspicious_errors,
    }
