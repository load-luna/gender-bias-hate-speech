from __future__ import annotations

from collections import defaultdict
from unittest import result

import numpy as np
from scipy.stats import bootstrap
from seaborn.utils import ci
from transformers import data


def group_confusion_rates(y_true, y_pred, group_labels):
    """Return TPR and FPR per group for binary classification."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group_labels = np.asarray(group_labels)

    result = defaultdict(dict)

    for group in np.unique(group_labels):
        mask = group_labels == group
        yt = y_true[mask]
        yp = y_pred[mask]

        pos = yt == 1
        neg = yt == 0

        result[group]["tpr"] = (
            float(((yp == 1) & pos).sum() / pos.sum())
            if pos.sum() else np.nan
        )
        result[group]["fpr"] = (
            float(((yp == 1) & neg).sum() / neg.sum())
            if neg.sum() else np.nan
        )

    return dict(result)


def equalized_odds_gaps(y_true, y_pred, group_labels) -> dict[str, float]:
    """Return TPR and FPR gaps across groups."""
    rates = group_confusion_rates(y_true, y_pred, group_labels)
    tprs = [v["tpr"] for v in rates.values() if not np.isnan(v["tpr"])]
    fprs = [v["fpr"] for v in rates.values() if not np.isnan(v["fpr"])]

    return {
        "tpr_gap": float(max(tprs) - min(tprs)) if len(tprs) >= 2 else np.nan,
        "fpr_gap": float(max(fprs) - min(fprs)) if len(fprs) >= 2 else np.nan,
    }

# Mean Direction of Change
#positiv when probability increased after swap
#negative when probability decreased after swap
def mean_change(original_probs, swapped_probs, axis=-1):
    return np.mean(swapped_probs - original_probs, axis=axis)

# Absolute difference between original and swapped predictions
def ctf_gap(original_probs, swapped_probs, axis=-1):
    return np.mean(np.abs(swapped_probs - original_probs), axis=axis)

def compute_paired_bootstrap_confidence_interval(paired_predictions, label_id, statistic_func, n_resamples=5000, confidence_level=0.95):
    """Compute paired bootstrap confidence interval for a given statistic."""
    if len(paired_predictions) == 0:
        raise ValueError("Cannot compute confidence interval for an empty dataset.")

    original_scores = paired_predictions[
        f"probability_{label_id}_original"
    ].to_numpy()

    swapped_scores = paired_predictions[
        f"probability_{label_id}_swapped"
    ].to_numpy()

    data = (original_scores, swapped_scores)

    result = bootstrap(
        data=data,
        statistic=statistic_func,
        paired=True,
        vectorized=True,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        method="BCa",
        batch=100,
        rng=np.random.default_rng(42),
    )

    estimate = statistic_func(original_scores, swapped_scores)
    ci = result.confidence_interval

    return {"estimate": estimate, "confidence_interval": ci}



