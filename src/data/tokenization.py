"""Dataset tokenization and pair-aware sequence-length filtering."""

from datasets import Dataset, DatasetDict

from src.data.gender_labeling import swap_gender


_ORIGINAL_LENGTH = "_pair_filter_original_length"
_SWAPPED_LENGTH = "_pair_filter_swapped_length"
_SWAP_CHANGED = "_pair_filter_swap_changed"
_KEEP_EXAMPLE = "_pair_filter_keep"


def tokenize_texts(
    texts, tokenizer, *, truncation=False, max_length=512, padding="max_length"
):
    """Tokenize one text or a batch with shared project options."""
    return tokenizer(
        texts, truncation=truncation, padding=padding, max_length=max_length
    )

#outputs config into workable format
def resolve_tokenization_options(config):
    """Convert the configured length policy to tokenizer keyword arguments."""
    policy = config["overlength_policy"]
    if policy not in {"filter_pairs", "truncate"}:
        raise ValueError(
            "tokenization.overlength_policy must be 'filter_pairs' or 'truncate'"
        )
    return {"max_length": config["max_length"], "truncation": policy == "truncate"}


def tokenize_dataset(dataset, tokenizer, truncation=False, max_length=512, padding="max_length"):
    """Tokenize text and optional CLP swap text in one shared dataset pass."""
    def preprocess_function(examples):
        encoded = tokenize_texts(
            examples["text"], tokenizer,
            truncation=truncation, padding=padding, max_length=max_length,
        )
        if "clp_text" in examples:
            swapped = tokenize_texts(
                examples["clp_text"], tokenizer,
                truncation=truncation, padding=padding, max_length=max_length,
            )
            encoded.update({f"clp_{key}": value for key, value in swapped.items()})
            encoded["clp_valid"] = [
                bool(candidate and original_ids != swapped_ids)
                for candidate, original_ids, swapped_ids in zip(
                    examples["clp_pair_candidate"],
                    encoded["input_ids"],
                    swapped["input_ids"],
                )
            ]
        return encoded

    return dataset.map(preprocess_function, batched=True)


def check_max_sequence_length(tokenizer, tokenized_dataset):
    """Count rows whose existing input IDs exceed the tokenizer's model limit."""
    maximum = tokenizer.model_max_length
    splits = (
        tokenized_dataset.values()
        if isinstance(tokenized_dataset, DatasetDict)
        else (tokenized_dataset,)
    )
    return sum(
        len(example["input_ids"]) > maximum
        for split in splits
        for example in split
    )


def filter_pair_length_eligible(dataset, tokenizer, *, max_length=512):
    """Keep rows only when the original and changed gender swap both fit.

    Apply to the fixed splits before mitigation so every experiment condition
    uses the same eligible rows. No truncation is used during this check.
    """
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("max_length must be a positive integer")

    if isinstance(dataset, DatasetDict):
        filtered_splits = {}
        split_reports = {}
        for split_name, split in dataset.items():
            filtered_splits[split_name], split_reports[split_name] = (
                filter_pair_length_eligible(split, tokenizer, max_length=max_length)
            )
        totals = {
            key: sum(report[key] for report in split_reports.values())
            for key in (
                "n_before", "n_after", "n_removed", "n_original_too_long",
                "n_removed_only_because_swap_too_long", "n_changed_swaps_too_long",
                "n_pairs_checked",
            )
        }
        return DatasetDict(filtered_splits), {
            "method": "joint_original_swap_max_length_filter",
            "max_length": max_length,
            **totals,
            "splits": split_reports,
        }

    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be a datasets.Dataset or DatasetDict")
    if "text" not in dataset.column_names:
        raise ValueError("dataset must contain a 'text' column")

    def measure_lengths(batch):
        original_texts = batch["text"]
        swapped_texts = [swap_gender(text) for text in original_texts]
        original_tokens = tokenize_texts(
            original_texts, tokenizer, truncation=False,
            max_length=max_length, padding=False,
        )["input_ids"]
        swapped_tokens = tokenize_texts(
            swapped_texts, tokenizer, truncation=False,
            max_length=max_length, padding=False,
        )["input_ids"]
        original_lengths = [len(ids) for ids in original_tokens]
        swapped_lengths = [len(ids) for ids in swapped_tokens]
        changed = [original != swapped for original, swapped
                   in zip(original_texts, swapped_texts)]
        keep = [
            original_length <= max_length and
            (not did_change or swapped_length <= max_length)
            for original_length, did_change, swapped_length
            in zip(original_lengths, changed, swapped_lengths)
        ]
        return {
            _ORIGINAL_LENGTH: original_lengths,
            _SWAPPED_LENGTH: swapped_lengths,
            _SWAP_CHANGED: changed,
            _KEEP_EXAMPLE: keep,
        }

    measured = dataset.map(
        measure_lengths,
        batched=True,
        desc=f"Check original/swap lengths (max_length={max_length})",
    )
    original_lengths = measured[_ORIGINAL_LENGTH]
    swapped_lengths = measured[_SWAPPED_LENGTH]
    changed = measured[_SWAP_CHANGED]
    keep = measured[_KEEP_EXAMPLE]
    original_too_long = [length > max_length for length in original_lengths]
    changed_swap_too_long = [
        did_change and length > max_length
        for did_change, length in zip(changed, swapped_lengths)
    ]
    removed_only_for_swap = [
        not keep_row and not original_long and swap_long
        for keep_row, original_long, swap_long
        in zip(keep, original_too_long, changed_swap_too_long)
    ]

    filtered = measured.filter(lambda row: row[_KEEP_EXAMPLE])
    filtered = filtered.remove_columns([
        _ORIGINAL_LENGTH, _SWAPPED_LENGTH, _SWAP_CHANGED, _KEEP_EXAMPLE,
    ])
    return filtered, {
        "n_before": len(dataset),
        "n_after": len(filtered),
        "n_removed": len(dataset) - len(filtered),
        "n_original_too_long": int(sum(original_too_long)),
        "n_changed_swaps_too_long": int(sum(changed_swap_too_long)),
        "n_removed_only_because_swap_too_long": int(sum(removed_only_for_swap)),
        "n_pairs_checked": int(sum(changed)),
    }
