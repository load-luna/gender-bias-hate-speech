"""Descriptive reports for dataset splits and held-out gender-term groups."""

from collections.abc import Mapping

import pandas as pd

from src.data.gender_labeling import (
    gender_labels,
    gender_lexicon,
    get_words,
    normalize_gender_pairs,
)


HELD_OUT_CATEGORIES = (
    "held_out_male_only",
    "held_out_female_only",
    "held_out_both_sides",
    "other_gender_terms_no_held_out",
    "no_gender_terms",
)


def _config_dict(config):
    return config.to_dict() if hasattr(config, "to_dict") else config


def _as_frame(dataset):
    if isinstance(dataset, pd.DataFrame):
        return dataset
    to_pandas = getattr(dataset, "to_pandas", None)
    if callable(to_pandas):
        return to_pandas()
    # Also accept ordinary split data represented as records or as a
    # column-oriented mapping. This is useful when a caller has converted a
    # DatasetDict to plain Python objects before requesting the report.
    if isinstance(dataset, (Mapping, list, tuple)):
        try:
            return pd.DataFrame(dataset)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Could not convert split data to a pandas DataFrame; "
                f"received {type(dataset).__name__}."
            ) from exc
    raise TypeError(
        "Each split must be a pandas DataFrame, a Hugging Face Dataset, "
        "or row/column data convertible to a DataFrame; "
        f"received {type(dataset).__name__}."
    )


def _get_splits(dataset):
    """Return named splits; treat a single Dataset/DataFrame as split 'all'."""
    # DatasetDict is also mapping-like and exposes ``column_names`` (as a
    # mapping of split names to columns), so detect split collections first.
    if isinstance(dataset, Mapping):
        if not dataset:
            raise ValueError("At least one dataset split is required.")
        return [
            (str(name), _as_frame(split))
            for name, split in dataset.items()
        ]
    if isinstance(dataset, pd.DataFrame) or hasattr(dataset, "column_names"):
        return [("all", _as_frame(dataset))]
    raise TypeError("dataset must be a split mapping, Dataset, or DataFrame.")


def _label_names(config):
    raw_names = config["dataset"]["label_mapping"]["names"]
    return {int(class_id): str(name) for class_id, name in raw_names.items()}


def count_label_distribution(dataset, config, *, split_name="all"):
    """Create a long-form gender-label by task-label count table.

    Percentages are included with explicit denominators: share of the split,
    share within the gender group, and share within the task class.
    """
    config = _config_dict(config)
    label_names = _label_names(config)
    frame = _as_frame(dataset)
    required = {"label", "gender_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["label"] = frame["label"].astype(int)
    frame["gender_label"] = frame["gender_label"].astype(int)
    unknown_classes = set(frame["label"]) - set(label_names)
    unknown_genders = set(frame["gender_label"]) - set(gender_labels)
    if unknown_classes:
        raise ValueError(f"Dataset contains unknown task labels: {sorted(unknown_classes)}")
    if unknown_genders:
        raise ValueError(f"Dataset contains unknown gender labels: {sorted(unknown_genders)}")

    split_size = len(frame)
    gender_totals = frame["gender_label"].value_counts().to_dict()
    class_totals = frame["label"].value_counts().to_dict()
    rows = []

    for gender_id, gender_name in gender_labels.items():
        for class_id, class_name in label_names.items():
            n = int(((frame["gender_label"] == gender_id) &
                     (frame["label"] == class_id)).sum())
            rows.append({
                "split": split_name,
                "gender_label_id": gender_id,
                "gender_label": gender_name,
                "class_id": class_id,
                "class_name": class_name,
                "n": n,
                "pct_of_split": 100.0 * n / split_size if split_size else None,
                "pct_within_gender_label": (
                    100.0 * n / gender_totals[gender_id]
                    if gender_totals.get(gender_id, 0) else None
                ),
                "pct_within_class_label": (
                    100.0 * n / class_totals[class_id]
                    if class_totals.get(class_id, 0) else None
                ),
            })
    return pd.DataFrame(rows)


def _held_out_category(text, male_terms, female_terms, all_gender_terms):
    words = set(get_words(str(text)))
    has_male_side = bool(words & male_terms)
    has_female_side = bool(words & female_terms)

    if has_male_side and has_female_side:
        return "held_out_both_sides"
    if has_male_side:
        return "held_out_male_only"
    if has_female_side:
        return "held_out_female_only"
    if words & all_gender_terms:
        return "other_gender_terms_no_held_out"
    return "no_gender_terms"


def count_held_out_distribution(dataset, config, held_out_pairs, *, split_name="all"):
    """Count disjoint held-out presence categories by task class.

    A text is assigned to exactly one category. If multiple configured pairs
    are used, presence on each side is pooled across those pairs.
    """
    config = _config_dict(config)
    label_names = _label_names(config)
    pairs = normalize_gender_pairs(held_out_pairs)
    if not pairs:
        raise ValueError("held_out_pairs must contain at least one valid pair.")

    frame = _as_frame(dataset)
    required = {"text", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["label"] = frame["label"].astype(int)
    unknown_classes = set(frame["label"]) - set(label_names)
    if unknown_classes:
        raise ValueError(f"Dataset contains unknown task labels: {sorted(unknown_classes)}")

    male_terms = {male for male, _ in pairs}
    female_terms = {female for _, female in pairs}
    all_gender_terms = set(gender_lexicon["male"]) | set(gender_lexicon["female"])
    frame["_held_out_category"] = frame["text"].map(
        lambda text: _held_out_category(
            text, male_terms, female_terms, all_gender_terms
        )
    )

    split_size = len(frame)
    class_totals = frame["label"].value_counts().to_dict()
    category_totals = frame["_held_out_category"].value_counts().to_dict()
    rows = []
    for category in HELD_OUT_CATEGORIES:
        for class_id, class_name in label_names.items():
            n = int(((frame["_held_out_category"] == category) &
                     (frame["label"] == class_id)).sum())
            rows.append({
                "split": split_name,
                "held_out_category": category,
                "class_id": class_id,
                "class_name": class_name,
                "n": n,
                "pct_of_split": 100.0 * n / split_size if split_size else None,
                "pct_within_category": (
                    100.0 * n / category_totals[category]
                    if category_totals.get(category, 0) else None
                ),
                "pct_within_class_label": (
                    100.0 * n / class_totals[class_id]
                    if class_totals.get(class_id, 0) else None
                ),
            })
    return pd.DataFrame(rows)


def get_data_report(dataset, config, held_out_pairs=()):
    """Build split sizes and descriptive gender/task/held-out tables.

    The returned DataFrames are suitable for CSV export. Analyze the fixed
    splits after shared length filtering and before any training mitigation.
    """
    config = _config_dict(config)
    pairs = normalize_gender_pairs(held_out_pairs)
    gender_tables = []
    held_out_tables = []
    split_sizes = {}

    for split_name, split in _get_splits(dataset):
        split_sizes[split_name] = len(split)
        gender_tables.append(
            count_label_distribution(split, config, split_name=split_name)
        )
        if pairs:
            held_out_tables.append(
                count_held_out_distribution(
                    split, config, pairs, split_name=split_name
                )
            )

    return {
        "split_sizes": split_sizes,
        "held_out_pairs": [list(pair) for pair in pairs],
        "gender_class_distribution": pd.concat(gender_tables, ignore_index=True),
        "held_out_distribution": (
            pd.concat(held_out_tables, ignore_index=True)
            if held_out_tables else pd.DataFrame(columns=[
                "split", "held_out_category", "class_id", "class_name", "n",
                "pct_of_split", "pct_within_category", "pct_within_class_label",
            ])
        ),
    }
