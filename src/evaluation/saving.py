"""Portable per-example predictions, readable without loading the model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def prediction_frame(results):
    """Accept flat or grouped results; store one numeric column per class."""
    values = results["all"] if "all" in results else results
    columns = {"pair_ids": "pair_id", "texts": "text", "labels": "label",
               "predictions": "prediction", "gender_labels": "gender_label"}
    frame = pd.DataFrame({columns.get(key, key): value for key, value in values.items()
                          if key not in ("logits", "probabilities")})
    for key, prefix in (("logits", "logit"), ("probabilities", "probability")):
        array = np.asarray(values[key])
        if array.ndim != 2 or len(array) != len(frame):
            raise ValueError(f"{key} must align with prediction rows.")
        for class_id in range(array.shape[1]):
            frame[f"{prefix}_{class_id}"] = array[:, class_id]
    return frame


def save_predictions(results, path, *, label_names=None, metadata=None):
    """Save CSV plus class mapping and provenance JSON alongside it."""
    frame = prediction_frame(results)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {"schema_version": 1, "class_names": label_names,
            "pair_id_scope": "Position within the same test split",
            "metadata": metadata or {}}
    encoded = json.dumps(info, indent=2, ensure_ascii=False, allow_nan=False)
    frame.to_csv(path, index=False, encoding="utf-8")
    path.with_suffix(".metadata.json").write_text(encoded, encoding="utf-8")
    return path


def save_counterfactual_predictions(original, swapped, path):
    """Validate pairs and save original/swapped scores side by side."""
    before, after = prediction_frame(original), prediction_frame(swapped)
    for column in ("pair_id", "label"):
        if not np.array_equal(before[column], after[column]):
            raise ValueError(f"Original and swapped {column} do not align.")
    if before["pair_id"].duplicated().any():
        raise ValueError("pair_id must be unique.")
    if "original_text" not in after or not np.array_equal(before["text"], after["original_text"]):
        raise ValueError("Swapped source texts do not match original predictions.")
    frame = before.merge(after, on="pair_id", suffixes=("_original", "_swapped"), validate="one_to_one", sort=False)
    frame["prediction_flipped"] = frame["prediction_original"] != frame["prediction_swapped"]
    for column in before.columns:
        if column.startswith("probability_"):
            frame[f"delta_{column}"] = frame[f"{column}_swapped"] - frame[f"{column}_original"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    return path
