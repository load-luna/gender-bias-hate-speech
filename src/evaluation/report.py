"""Grouped single-label classification reports for binary and multiclass tasks."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def evaluate_groups(group_predictions, label_names, comparison=("female", "male")):
    """Use fixed class IDs; undefined rates are None, never artificial zeros.

    Macro F1 uses every configured class, with undefined F1 set to zero.
    Balanced accuracy averages recall over classes present in each group.
    Fairness differences compare comparison[0] minus comparison[1], per class
    versus all other classes. They do not include the overlapping 'all' group.
    """
    names = {int(key): str(value) for key, value in label_names.items()}
    ids = sorted(names)
    if len(ids) < 2:
        raise ValueError("At least two configured classes are required.")
    report = {
        "class_labels": [{"id": key, "name": names[key]} for key in ids],
        "definitions": {
            "f1_macro": "All configured classes; undefined F1 contributes zero.",
            "balanced_accuracy": "Mean recall over classes with positive support.",
            "undefined_rates": "null means denominator is zero.",
            "fairness": "Per-class one-vs-rest; signed differences are first minus second group.",
        },
        "groups": {},
    }
    for group, values in group_predictions.items():
        yt, yp = np.asarray(values["labels"]), np.asarray(values["predictions"])
        if yt.ndim != 1 or yp.ndim != 1 or yt.shape != yp.shape:
            raise ValueError(f"{group}: labels and predictions must be aligned 1D arrays.")
        if not np.isin(yt, ids).all() or not np.isin(yp, ids).all():
            raise ValueError(f"{group}: unknown class ID.")
        cm = confusion_matrix(yt, yp, labels=ids)
        n = len(yt)
        rows = []
        normalized = []
        for index, class_id in enumerate(ids):
            tp = int(cm[index, index])
            support = int(cm[index].sum())
            predicted = int(cm[:, index].sum())
            fn, fp = support - tp, predicted - tp
            tn = n - tp - fn - fp
            rows.append({
                "class_id": class_id, "class_name": names[class_id],
                "support": support, "negative_support": n - support,
                "predicted_count": predicted,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": _ratio(tp, predicted),
                "recall": _ratio(tp, support),
                "f1": _ratio(2 * tp, 2 * tp + fp + fn),
                "fpr": _ratio(fp, n - support),
            })
            normalized.append([_ratio(int(value), support) for value in cm[index]])
        recalls = [row["recall"] for row in rows if row["recall"] is not None]
        report["groups"][group] = {
            "summary": {
                "n": n,
                "accuracy": _ratio(int(np.trace(cm)), n),
                "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
                "f1_macro": float(f1_score(yt, yp, labels=ids, average="macro", zero_division=0)) if n else None,
                "f1_weighted": float(f1_score(yt, yp, labels=ids, average="weighted", zero_division=0)) if n else None,
            },
            "per_class": rows,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized_true": normalized,
        }
    first, second = comparison
    if first == second or "all" in comparison:
        raise ValueError("Compare two distinct subgroups, not the overall group.")
    gaps = []
    for index, class_id in enumerate(ids):
        row = {"class_id": class_id, "class_name": names[class_id]}
        for rate, key in (("recall", "tpr"), ("fpr", "fpr")):
            a = report["groups"].get(first, {}).get("per_class")
            b = report["groups"].get(second, {}).get("per_class")
            av = a[index][rate] if a else None
            bv = b[index][rate] if b else None
            delta = av - bv if av is not None and bv is not None else None
            row[f"{key}_difference"] = delta
            row[f"{key}_gap_abs"] = abs(delta) if delta is not None else None
        gaps.append(row)
    report["fairness"] = {"first_group": first, "second_group": second, "per_class": gaps}
    return report


def save_evaluation_report(report, output_dir):
    """Save one complete JSON, three flat CSV tables and a matrix overview."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    pd.DataFrame([
        {"group": group, **values["summary"]}
        for group, values in report["groups"].items()
    ]).to_csv(output_dir / "group_metrics.csv", index=False)
    pd.DataFrame([
        {"group": group, **row}
        for group, values in report["groups"].items()
        for row in values["per_class"]
    ]).to_csv(output_dir / "class_metrics.csv", index=False)
    pd.DataFrame([
        {"first_group": report["fairness"]["first_group"],
         "second_group": report["fairness"]["second_group"], **row}
        for row in report["fairness"]["per_class"]
    ]).to_csv(output_dir / "fairness_metrics.csv", index=False)

    names = [row["name"] for row in report["class_labels"]]
    side = max(5, len(names) * 0.75)
    fig, axes = plt.subplots(max(1, len(report["groups"])), 2,
                             figsize=(2 * side, side * max(1, len(report["groups"]))), squeeze=False)
    for row_index, (group, values) in enumerate(report["groups"].items()):
        for column, (key, title, fmt) in enumerate((
            ("confusion_matrix", "Counts", "d"),
            ("confusion_matrix_normalized_true", "Recall-normalized (undefined rows blank)", ".2f"),
        )):
            matrix = np.asarray(values[key], dtype=float if column else int)
            ConfusionMatrixDisplay(matrix, display_labels=names).plot(
                ax=axes[row_index, column], values_format=fmt, colorbar=False,
                xticks_rotation=45, cmap="Blues",
                im_kw={"vmin": 0, "vmax": 1} if column else None,
            )
            if column:
                for text in axes[row_index, column].texts:
                    if text.get_text() == "nan":
                        text.set_text("—")
            axes[row_index, column].set_title(f"{group} (n={values['summary']['n']}) — {title}")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return metrics_path
