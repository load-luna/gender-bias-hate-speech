import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix, 

)

def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None

def _difference(value1, value2):
    if value1 is None or value2 is None:
        return None
    return float(value1 - value2)

def compute_performance_metrics(results, class_ids):
    y_true = results["label"]
    y_pred = results["prediction"]

    return {
        "n"
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "f1_macro": float(
            f1_score(
                y_true, y_pred,
                labels=class_ids,
                average="macro",
                zero_division=np.nan,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true, y_pred,
                labels=class_ids,
                average="weighted",
                zero_division=np.nan,
            )
        ),
    }





def compute_performance_per_class(results, class_ids, label_names):
    rows = []
    y_true = results["label"]
    y_pred = results["prediction"]
    n = len(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    
    for index, class_id in enumerate(class_ids):
        
        support = int(cm[index].sum())
        negative_support = n - support
        predicted = int(cm[:, index].sum())
        tp = int(cm[index, index])
        fn = support - tp
        fp = predicted - tp
        tn = n - tp - fn - fp
        rows.append({
            "class_id": class_id, 
            "class_name": label_names[index],
            "support": support, 
            "negative_support": negative_support,
            "predicted_count": predicted,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": _ratio(tp, predicted),
            "recall": _ratio(tp, support),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "fpr": _ratio(fp, negative_support),
        })
    
    return rows



#to do test if tpr = recall
def compute_confusion_rates(results,class_ids, label_names):
    y_true= np.asarray(results["label"])
    y_pred= np.asarray(results["prediction"])

    confusion_rates = {}

    for i, class_name in zip(class_ids, label_names):

        pos = y_true == i
        neg = y_true != i

        tpr = _ratio(((y_pred==i) & pos).sum(),pos.sum())
        fpr = _ratio(((y_pred==i) & neg).sum(),neg.sum())

        confusion_rates[f"{class_name}"] ={
            "TPR":tpr,
            "FPR":fpr
        }

    return confusion_rates

#needs group_predictions(results)
def compute_group_gaps(grouped_results,class_ids, label_names, plus="female",minus="male"):
    plus_class = grouped_results[plus]
    minus_class = grouped_results[minus]

    plus_confusion_rates = compute_confusion_rates(plus_class,class_ids, label_names)
    minus_confusion_rates = compute_confusion_rates(minus_class,class_ids, label_names)

    gap = {}
    for label in plus_confusion_rates.keys():
        gap[label] = {
            "TPR_Gap" : _difference(plus_confusion_rates[label]["TPR"], minus_confusion_rates[label]["TPR"]),
            "FPR_Gap" : _difference(plus_confusion_rates[label]["FPR"], minus_confusion_rates[label]["FPR"]),
        }


    return gap

def group_predictions(results, gender_labels):
    """Select all prediction arrays using the same mask for each group."""
    gender = np.asarray(results["gender_label"])
    masks = {"all": np.ones(len(gender), dtype=bool)}
    for group_id, group_name in gender_labels.items():
        masks[group_name] = gender == group_id
    return {
        group: {key: np.asarray(value)[mask] for key, value in results.items()}
        for group, mask in masks.items()
    }


def eval_prediction(results, class_ids, label_names, gender_labels):
    """Return sample count, confusion matrix and report for all/gender groups.

    results is the dictionary returned by predict_dataset. class_ids and
    label_names must have matching order; gender_labels maps IDs to names.
    Empty groups return None for their matrix and report.
    """
    class_ids = list(class_ids)
    label_names = list(label_names)

    #test format
    if not class_ids or len(class_ids) != len(label_names):
        raise ValueError("Provide one label name for each class ID.")
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("Class IDs must be unique.")
    if not all(isinstance(name, str) for name in label_names):
        raise ValueError("Label names must be strings.")
    if len(set(label_names)) != len(label_names):
        raise ValueError("Label names must be unique.")
    group_names = list(gender_labels.values())
    if "all" in group_names or len(set(group_names)) != len(group_names):
        raise ValueError("Gender names must be unique and cannot be 'all'.")

    #reject inconsistent inputs
    labels = np.asarray(results["label"])
    predictions = np.asarray(results["prediction"])
    gender = np.asarray(results["gender_label"])
    if labels.ndim != 1 or predictions.shape != labels.shape or gender.shape != labels.shape:
        raise ValueError("Labels, predictions and gender labels must be aligned 1D arrays.")
    for key, value in results.items():
        array = np.asarray(value)
        if array.ndim == 0 or len(array) != len(labels):
            raise ValueError(f"{key} must have one row per example.")
    if not np.isin(labels, class_ids).all() or not np.isin(predictions, class_ids).all():
        raise ValueError("Labels or predictions contain unknown class IDs.")
    if not np.isin(gender, list(gender_labels)).all():
        raise ValueError("Results contain unknown gender IDs.")

    grouped_results = group_predictions(results, gender_labels)
    evaluation = {}
    for group_name, group_results in grouped_results.items():
        n_samples = len(group_results["label"])
        # An empty group has no measurable classification performance.
        evaluation[group_name] = {
            "n_samples": n_samples,
            "performance": None,
            "per_class": [],
            "confusion_rates": None,
            "confusion_matrix": None,
            "confusion_matrix_normalized_true": None,
        }

        if n_samples == 0:
            continue

        #performance // preformace over the whole group
        #per-class // preformace per class/label
        #confusion matrix // confusion matrix for the group
        #confusion matrix normalized // confusion matrix normalized for the group

        evaluation[group_name]["n_samples"] = n_samples

        evaluation[group_name]["performance"] = compute_performance_metrics(
            group_results, class_ids
        )
        evaluation[group_name]["per_class"] = compute_performance_per_class(
            group_results, class_ids, label_names
        )
        evaluation[group_name]["confusion_rates"] = compute_confusion_rates(
            group_results, class_ids, label_names
        )

        cm = confusion_matrix(
            group_results["label"], group_results["prediction"], labels=class_ids
        )

        normalized = [
            [_ratio(value, row.sum()) for value in row]
            for row in cm
        ]
        
        evaluation[group_name]["confusion_matrix"] = cm.tolist()
        evaluation[group_name]["confusion_matrix_normalized_true"] = normalized


    female_minus_male_gap = compute_group_gaps(
        grouped_results,
        class_ids,
        label_names,
        plus="female",
        minus="male",
    )

    return {
        "class_labels": [
            {"id": class_id, "name": name}
            for class_id, name in zip(class_ids, label_names)
        ],
        "groups": evaluation,
        "gaps": {
            "female_minus_male": female_minus_male_gap,
        },
    }
