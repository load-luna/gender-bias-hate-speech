import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def check_keys(dict, expected_keys):
    actual_keys = set(dict.keys())

    missing_keys = expected_keys - actual_keys
    unexpected_keys = actual_keys - expected_keys

    if missing_keys or unexpected_keys:
        raise ValueError(
            f"Invalid prediction keys. "
            f"Missing: {sorted(missing_keys)}. "
            f"Unexpected: {sorted(unexpected_keys)}."
        )   


def make_predictions_frame(predictions):
    #predictions key test
    expected_keys = {
        "id",
        "text",
        "label",
        "prediction",
        "gender_label",
        "logits",
        "probabilities",
    }

    check_keys(predictions, expected_keys)    

    #build dataframe

    #single values
    frame = pd.DataFrame({key: value for key, value in predictions.items()
                        if key not in ("logits", "probabilities")})

    #add array values
    for key, prefix in (("logits", "logit"), ("probabilities", "probability")):
            array = np.asarray(predictions[key])
            if array.ndim != 2 or len(array) != len(frame):
                raise ValueError(f"{key} must align with prediction rows.")
            for class_id in range(array.shape[1]):
                frame[f"{prefix}_{class_id}"] = array[:, class_id]

    return frame



def save_predictions(predictions,output_path, *, label_names=None, metadata=None):
    frame = make_predictions_frame(predictions)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    info = {"class_names": label_names,
            "id_scope": "Position within the same test split",
            "metadata": metadata or {}}
    encoded = json.dumps(info, indent=2, ensure_ascii=False, allow_nan=False)
    frame.to_csv(path, index=False, encoding="utf-8")
    path.with_suffix(".metadata.json").write_text(encoded, encoding="utf-8")

    return path


def save_counterfactual_predictions(original, swapped, path, *, label_names=None, metadata=None):
    """Validate pairs and save original/swapped scores side by side."""
    before, after = make_predictions_frame(original), make_predictions_frame(swapped)
    for column in ("id", "label"):
        if not np.array_equal(before[column], after[column]):
            raise ValueError(f"Original and swapped {column} do not align.")
    if before["id"].duplicated().any() or after["id"].duplicated().any():
        raise ValueError("id must be unique.")

    frame = before.merge(after, on="id", suffixes=("_original", "_swapped"), validate="one_to_one", sort=False)
    frame["prediction_flipped"] = frame["prediction_original"] != frame["prediction_swapped"]
    frame["gender_flipped"] = frame["gender_label_original"] != frame["gender_label_swapped"]
    for column in before.columns:
        if column.startswith("probability_"):
            frame[f"delta_{column}"] = frame[f"{column}_swapped"] - frame[f"{column}_original"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    info = {"class_names": label_names,
                "id_scope": "Position within the same test split",
                "metadata": metadata or {}}
    encoded = json.dumps(info, indent=2, ensure_ascii=False, allow_nan=False)
    frame.to_csv(path, index=False, encoding="utf-8")
    path.with_suffix(".metadata.json").write_text(encoded, encoding="utf-8")
    return path





def save_evaluation(evaluation, output_dir, *, label_names=None, metadata=None):    
    """Save one complete JSON, 3 flat CSV tables and a matrix overview."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    pd.DataFrame([
        {
            "n_samples": values["n_samples"],
            "group": group,
            **(values["performance"] or {})
        }
        for group, values in evaluation["groups"].items()
    ]).to_csv(output_dir / "performance_metrics.csv", index=False)

    pd.DataFrame([
        {
            "n_samples": values["n_samples"],
            "group": group,
            **(row or {})
        }
        for group, values in evaluation["groups"].items()
        for row in values["per_class"]
    ]).to_csv(output_dir / "class_metrics.csv", index=False)

    pd.DataFrame([
            {
                "group": group, 
                "class_name" : class_name,
                **row,
            }
            for group, values in evaluation["groups"].items()
            for class_name, row in (values["confusion_rates"] or {}).items()
        ]).to_csv(output_dir / "confusion_rates.csv", index=False)

    pd.DataFrame([
        {
                "class_name" : class_name,
                **row,
        }
        for class_name, row in (evaluation["gaps"]["female_minus_male"] or {}).items()
    ]).to_csv(output_dir / "female_minus_male_gaps.csv", index=False)


    names = [row["name"] for row in evaluation["class_labels"]]
    side = max(5, len(names) * 0.75)
    fig, axes = plt.subplots(max(1, len(evaluation["groups"])), 2,
                             figsize=(2 * side, side * max(1, len(evaluation["groups"]))), squeeze=False)
    for row_index, (group, values) in enumerate(evaluation["groups"].items()):
        n_samples = values["n_samples"]
        if n_samples == 0:
            for ax in axes[row_index]:
                ax.set_axis_off()
                ax.set_title(f"{group} (n=0)")
                ax.text(
                    0.5, 0.5,
                    "Keine Beispiele",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
            )
            continue

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
            axes[row_index, column].set_title(f"{group} (n={n_samples}) — {title}")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


    return metrics_path



def _json_values(value):
    """Convert nested NumPy values and undefined numbers without changing input."""
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_values(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    if isinstance(value, np.generic):
        return _json_values(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_counterfactual_eval(evaluation, output_dir, *, metadata=None):
    """Save counterfactual groups as JSON and a flat summary CSV.

    Matrices remain in JSON with their axis labels. Empty-group rates are null.
    Existing files with these names are overwritten. Returns the JSON path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    matrix_keys = {"prediction_swap_matrix", "gender_label_swap_matrix"}
    # Build serializable copies before writing any files.
    groups = _json_values(evaluation)
    report = {"groups": groups, "metadata": _json_values(metadata or {})}
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    metrics_path = output_dir / "counterfactual_metrics.json"
    metrics_path.write_text(encoded, encoding="utf-8")

    summary = pd.DataFrame([
        {"group": group, **{
            key: value for key, value in values.items() if key not in matrix_keys
        }}
        for group, values in groups.items()
    ])
    summary.to_csv(
            output_dir / "counterfactual_summary.csv", index=False, encoding="utf-8"
    )

    matrix_dir = output_dir / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    #prediction_swap_matrix
    for group_name in ("all", "male_to_female", "female_to_male"):
        matrix = evaluation[group_name]["prediction_swap_matrix"]

        class_ids = matrix["class_ids"]
        label_names = matrix["label_names"]
        names = [label_names[class_id] for class_id in class_ids]

        frame = pd.DataFrame(
            matrix["transition_matrix"]["counts"],
            index=names,
            columns=names,
        )

        frame.index.name = "original_prediction"
        frame.columns.name = "swapped_prediction"    
        frame.to_csv(matrix_dir/ f"prediction_swap__{group_name}.csv")

    #gender_label_swap
    matrix = evaluation["all"]["gender_label_swap_matrix"]

    class_ids = matrix["class_ids"]
    label_names = matrix["label_names"]
    names = [label_names[class_id] for class_id in class_ids]

    frame = pd.DataFrame(
        matrix["transition_matrix"]["counts"],
        index=names,
        columns=names,
    )

    frame.index.name = "original_label"
    frame.columns.name = "swapped_label"    
    frame.to_csv(matrix_dir/ f"gender_label_swap.csv")





    return metrics_path
