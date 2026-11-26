import pandas as pd

from src.data.gender_labeling import (
    add_gender_label,
    contains_gender_pair_term,
    swap_gender_terms_dataset_with_exclusion,
    check_terms_after_exclusion
)



def selective_cda(
    train_dataset,
    target_gender=(0, 1),
    exclude_grammar_uncertain=True,
    seed=42,
    held_out_pairs=(),
):
    report = {
        "method": "selective_cda",
        "seed": seed,
        "target_gender": list(target_gender),
        "exclude_grammar_uncertain": exclude_grammar_uncertain,
        "n_before": len(train_dataset),
        "held_out_pairs": held_out_pairs,
        "groups": [],
    }
    train_dataset = train_dataset.copy()
    train_dataset["id"] = train_dataset["id"].astype("string")
        
    selected_by_gender = train_dataset["gender_label"].isin(target_gender)
    candidates = train_dataset.loc[selected_by_gender].copy()

    should_swap = candidates.apply(
        lambda row: check_terms_after_exclusion(
            row,
            column_name="text",
            exclude_grammar_uncertain=exclude_grammar_uncertain,
            exclude_pairs=held_out_pairs,
        ),
        axis=1,
    )

    selected = candidates.loc[should_swap].copy()

    report["n_candidates"] = len(candidates)
    report["n_selected"] = len(selected)
    report["excluded"] = len(candidates) - len(selected)


    if not selected.empty:
        swapped_dataset = selected.apply(
            swap_gender_terms_dataset_with_exclusion,
            axis=1,
            exclude_grammar_uncertain=exclude_grammar_uncertain,
            exclude_pairs=held_out_pairs,
        )
        swapped_dataset = swapped_dataset.apply(add_gender_label, axis=1)
    else:
        swapped_dataset = selected.iloc[0:0].copy()


    #id neu setzen & alte id speichern
    # ID des zugehörigen Originals speichern
    swapped_dataset["source_id"] = swapped_dataset["id"].astype("string")

    # Für jede Kopie eine eigene ID erzeugen
    swapped_dataset["id"] = (
        swapped_dataset["source_id"].astype("string") + "_swap"
    )
    augmented_dataset = pd.concat([swapped_dataset,train_dataset], ignore_index=True)

    if not augmented_dataset["id"].is_unique:
        raise ValueError("Die IDs im augmentierten Datensatz sind nicht eindeutig.")

    #neu mischen
    augmented_dataset = augmented_dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Count by the actual gender label after augmentation, not the source gender.
    before_counts = train_dataset.groupby(["label", "gender_label"]).size()
    after_counts = augmented_dataset.groupby(["label", "gender_label"]).size()
    for label, gender in before_counts.index.union(after_counts.index):
        n_before = int(before_counts.get((label, gender), 0))
        n_after = int(after_counts.get((label, gender), 0))
        report["groups"].append({
            "class_id": int(label),
            "gender_label": int(gender),
            "n_before": n_before,
            "n_after": n_after,
            "n_added": n_after - n_before,
        })

    report["n_after"] = len(augmented_dataset)
    report["n_added"] = len(swapped_dataset)


    return augmented_dataset, report
