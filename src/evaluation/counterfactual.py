"""Aligned lexical gender swaps and paired classification comparisons."""
import numpy as np
import pandas as pd

from src.data.gender_labeling import label_gender, swap_gender
from src.evaluation.report import evaluate_groups


def create_swapped_test_dataset(dataset):
    """Keep order and gold labels; record source text/group and actual changes.

    Pass the untokenized test split. Synthetic labels assume label invariance;
    lexical swaps are not guaranteed to preserve grammar or meaning.
    """
    def swap(example, index):
        original = example["text"]
        swapped = swap_gender(original)
        return {
            "pair_id": index,
            "original_text": original,
            "original_gender_label": label_gender(original),
            "text": swapped,
            "gender_label": label_gender(swapped),
            "swap_changed": original != swapped,
        }

    return dataset.map(swap, with_indices=True)


def evaluate_counterfactual_pairs(swapped, original_predictions, swapped_predictions, label_names):
    """Compare aligned predictions, with group membership fixed before swapping.

    Reports on changed rows avoid dilution by unswapped examples. Empty groups
    have null rates. Transition matrices use original -> swapped predictions.
    """
    labels = np.asarray(swapped["label"])
    before, after = np.asarray(original_predictions), np.asarray(swapped_predictions)
    if before.ndim != 1 or after.ndim != 1 or before.shape != labels.shape or after.shape != labels.shape:
        raise ValueError("Predictions must align one-to-one with the swapped test rows.")
    ids = sorted(int(key) for key in label_names)
    if not np.isin(before, ids).all() or not np.isin(after, ids).all():
        raise ValueError("Unknown predicted class ID.")
    changed = np.asarray(swapped["swap_changed"], dtype=bool)
    gender = np.asarray(swapped["original_gender_label"])
    masks = {"all": np.ones(len(labels), dtype=bool), "male": gender == 0,
             "female": gender == 1, "equal": gender == 2, "none": gender == 3}
    summaries = []
    for name, group in masks.items():
        mask = group & changed
        n = int(mask.sum())
        flips = before[mask] != after[mask]
        summaries.append({
            "original_group": name, "n_total": int(group.sum()), "n_changed": n,
            "coverage": float(n / group.sum()) if group.any() else None,
            "n_prediction_flips": int(flips.sum()),
            "flip_rate_changed": float(flips.mean()) if n else None,
            "n_correct_to_wrong": int(((before == labels) & (after != labels) & mask).sum()),
            "n_wrong_to_correct": int(((before != labels) & (after == labels) & mask).sum()),
            "prediction_transitions": [
                [int((mask & (before == source) & (after == target)).sum()) for target in ids]
                for source in ids
            ],
        })
    reports = {}
    for name, predictions in (("original_changed", before), ("swapped_changed", after)):
        reports[name] = evaluate_groups({
            group: {"labels": labels[mask & changed], "predictions": predictions[mask & changed]}
            for group, mask in masks.items()
        }, label_names)
    pairs = pd.DataFrame({
        "pair_id": swapped["pair_id"], "original_text": swapped["original_text"],
        "swapped_text": swapped["text"], "label": labels,
        "original_gender_label": gender, "swapped_gender_label": swapped["gender_label"],
        "swap_changed": changed, "original_prediction": before,
        "swapped_prediction": after, "prediction_flipped": before != after,
    })
    return {
        "definitions": {
            "groups": "Original gender group; male=0, female=1, equal=2, none=3.",
            "flip_rate": "Prediction changes among text-changed pairs only.",
            "transitions": "Rows=original predictions, columns=swapped predictions, class_ids order.",
            "label_assumption": "Original gold labels retained; lexical swaps need manual semantic review.",
        },
        "class_ids": ids, "groups": summaries,
    }, reports, pairs
