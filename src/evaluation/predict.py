"""Predict once and select gender groups without additional model calls."""
import numpy as np


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("Expected logits with shape (n_samples, n_classes >= 2).")
    if not np.isfinite(logits).all():
        raise ValueError("Logits must be finite.")
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def group_predictions(values):
    """Select every aligned array, including IDs and texts."""
    gender = np.asarray(values["gender_labels"])
    masks = {"all": np.ones(len(gender), dtype=bool), 
             "male": gender == 0,
             "female": gender == 1, 
             "equal": gender == 2, 
             "none": gender == 3}
    return {group: {key: np.asarray(value)[mask] for key, value in values.items()}
            for group, mask in masks.items()}


def predict_dataset(dataset, trainer):
    """Return flat arrays. Keep text during tokenization; IDs are split-local."""
    if not len(dataset):
        raise ValueError("Cannot predict an empty dataset.")
    logits, labels, _ = trainer.predict(dataset)
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    probabilities = softmax(logits)
    labels = np.asarray(labels)
    if labels.shape != (len(dataset),) or logits.shape[0] != len(dataset):
        raise ValueError("Predictions and gold labels must align with dataset rows.")
    values = {
        "predictions": logits.argmax(axis=1), "labels": labels,
        "logits": logits, "probabilities": probabilities,
        "gender_labels": np.asarray(dataset["gender_label"]),
        "texts": np.asarray(dataset["text"]),
        "pair_ids": np.asarray(dataset["pair_id"]) if "pair_id" in dataset.column_names else np.arange(len(dataset)),
    }
    for column in ("example_id", "source_row", "original_text", "original_gender_label", "swap_changed"):
        if column in dataset.column_names:
            values[column] = np.asarray(dataset[column])
    return values


def predict_dataset_withmask(dataset, trainer):
    """Return all/male/female/equal/none results from one inference pass."""
    return group_predictions(predict_dataset(dataset, trainer))
