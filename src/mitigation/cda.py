import pandas as pd

from src.data.gender_labeling import (
    add_gender_label,
    contains_gender_pair_term,
    swap_gender_terms_dataset,
)



def selective_cda(
    train_dataset,
    target_gender=(0, 1),
    exclude_grammar_uncertain=True,
    seed=42,
    exclude_gender_term_pairs=(),
):
    held_out = train_dataset["text"].map(
        lambda text: contains_gender_pair_term(text, exclude_gender_term_pairs)
    ) if exclude_gender_term_pairs else pd.Series(False, index=train_dataset.index)
    report = {
        "method": "selective_cda",
        "seed": seed,
        "target_gender": list(target_gender),
        "exclude_grammar_uncertain": exclude_grammar_uncertain,
        "n_before": len(train_dataset),
        "n_held_out_unchanged": int(held_out.sum()),
        "groups": [],
    }
    train_dataset = train_dataset.copy()
    train_dataset["id"] = train_dataset["id"].astype("string")
    
    selected_by_gender = train_dataset["gender_label"].isin(target_gender)
    selected = train_dataset[selected_by_gender & ~held_out].copy()

    

    #swap Gender Terms in Texts
    report["n_selected"] = len(selected)
    report["n_excluded_held_out_from_cda"] = int((selected_by_gender & held_out).sum())
    swapped_dataset = selected.copy()
    if not selected.empty:
        swapped_dataset = selected.apply(swap_gender_terms_dataset,axis=1)

    #add new Gender Lable - will be replaced!
        swapped_dataset = swapped_dataset.apply(add_gender_label,axis=1)

    #exclude grammar uncertain
    report["n_excluded_grammar_uncertain"] = 0
    if exclude_grammar_uncertain and not swapped_dataset.empty:
        report["n_excluded_grammar_uncertain"] = int(
            swapped_dataset["swap_grammar_uncertain"].sum()
        )
        swapped_dataset = swapped_dataset[
            ~swapped_dataset["swap_grammar_uncertain"]
        ].copy()


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
