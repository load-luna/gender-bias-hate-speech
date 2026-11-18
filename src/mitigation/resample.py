import random
import pandas as pd

#dublicat plus sample the rest
def over_sample(dataset,sample_length,seed=42):

    dataset_size = len(dataset)
    missing = sample_length - dataset_size

    if missing<0:
        raise ValueError("dataset must be smaller than sample length")
    if dataset_size == 0:
        raise ValueError("dataset is empty")
    if missing==0:
            return dataset.copy()

    
    mult, mod = divmod(missing, dataset_size)

    sample = dataset.sample(
        n=mod,
        replace=False,
        random_state=seed,
    )

    parts = [dataset] * (mult + 1) + [sample]

    return pd.concat(parts, ignore_index=True)



def down_sample(dataset,sample_length,seed=42):
    random.seed(seed)
    dataset_size = len(dataset)
    missing = dataset_size - sample_length

    if missing<0:
        raise ValueError("dataset must be bigger than sample length")
    if dataset_size == 0:
        raise ValueError("dataset is empty")
    if missing==0:
            return dataset.copy()

    sample = dataset.sample(
        n=sample_length,
        replace=False,
        random_state=seed,
    )

    return sample



def balance_gender_within_class(train_dataset, class_id, target_gender,strategy= "mean", seed=42):
    report = {
        "method": "balance_gender_within_class",
        "sampling_strategy" : strategy,
        "seed": seed,
        "target_gender": list(target_gender),
        "n_before": len(train_dataset),
        "groups": [],
    }   

    if not target_gender:
        raise ValueError("target_gender darf nicht leer sein.")

    if strategy not in {"mean", "under", "over"}:
        raise ValueError(f"strategy : {strategy} not defined")
    
    # train_dataset ist hier ein Pandas-DataFrame
    balanced_parts = []

    for label in class_id:
        class_subset = train_dataset[
            train_dataset["label"] == label
        ]

        selected = class_subset["gender_label"].isin(target_gender)

        gender_subset = class_subset[selected]
        rest_subset = class_subset[~selected]

        balanced_parts.append(rest_subset)

        # Anzahl pro ausgewählter Gender-Gruppe innerhalb dieser Klasse
        group_sizes = (
            gender_subset["gender_label"]
            .value_counts()
            .reindex(target_gender, fill_value=0)
        )

        if group_sizes.eq(0).any():
            raise ValueError(
                f"Klasse {label}: Mindestens eine Zielgruppe enthält keine Beispiele."
            )

        if strategy == "mean":
            target_size, remainder = divmod(
                int(group_sizes.sum()), len(target_gender)
            )
        elif strategy == "under":
            target_size = int(group_sizes.min())
            remainder = 0
        else:  # strategy == "over"
            target_size = int(group_sizes.max())
            remainder = 0

        for i, gender in enumerate(target_gender):
            subset = gender_subset[
                gender_subset["gender_label"] == gender
            ]

            # Rest auf die ersten Gruppen verteilen
            group_target = target_size + int(i < remainder)

            if len(subset) > group_target:
                sample = down_sample(subset, group_target, seed)
            elif len(subset) < group_target:
                sample = over_sample(subset, group_target, seed)
            else:
                sample = subset.copy()

            balanced_parts.append(sample)

            n_before = len(subset)
            n_after = len(sample)

            report["groups"].append({
                "class_id": int(label),
                "gender_label": int(gender),
                "n_before": n_before,
                "n_after": n_after,
                "n_added": max(0, n_after - n_before),
                "n_removed": max(0, n_before - n_after),        
            })

    #zusammenfügen und mischen
    balanced_set = pd.concat(balanced_parts, ignore_index=True)
    balanced_set = balanced_set.sample(frac=1, random_state=seed).reset_index(drop=True)

    #id nach sampeling neu setzen & alte id speichern
    balanced_set["source_id"] = balanced_set["id"]
    balanced_set["id"] = [
        f"train_balanced_{i}" for i in range(len(balanced_set))
    ]

    report["n_after"] = len(balanced_set)
    report["n_added"] = sum(group["n_added"] for group in report["groups"])
    report["n_removed"] = sum(group["n_removed"] for group in report["groups"])

    return balanced_set, report

