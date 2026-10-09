from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(
    y_true,
    y_pred,
    average: str = "macro",
) -> dict[str, float]:
    """Compute common classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average)),
    }


def tpr_gap(y_true, y_pred, group_labels) -> float:
    """Compute max-min True Positive Rate across groups.

    This starter assumes binary classification and a binary grouping
    variable. Extend explicitly if your final analysis uses 3+ gender groups.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group_labels = np.asarray(group_labels)

    groups = np.unique(group_labels)
    tprs = []

    for group in groups:
        mask = group_labels == group
        positives = y_true[mask] == 1
        denom = positives.sum()
        if denom == 0:
            continue
        tprs.append(float(((y_pred[mask] == 1) & positives).sum() / denom))

    if len(tprs) < 2:
        return float("nan")

    return float(max(tprs) - min(tprs))
