from __future__ import annotations

from collections import defaultdict

import numpy as np


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
