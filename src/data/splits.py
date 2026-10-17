"""Persist reusable text-grouped splits and record their identity per run."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import io
import json
import os

import numpy as np
from datasets import DatasetDict
from sklearn.model_selection import GroupShuffleSplit


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(data):
    return json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")


def load_or_create_splits(dataset, config, run_dir, *, copy_assignments=False):
    """Create a version once, or validate and reuse it without reshuffling.

    Call on the prepared/cleaned dataset, before augmentation or balancing.
    Equal text is kept together, including conflicting labels (none discarded).
    Fractions apply to text groups, so row fractions are approximate. This
    method is not stratified. Content-derived IDs identify text/label records;
    identical records receive occurrence suffixes. Reordering input is safe.
    Changing text, labels, seed or proportions requires a new version directory.
    Paths in config are relative to the current project working directory.
    """
    ds_config = config["dataset"]
    settings = ds_config["split"]
    directory = Path(settings["directory"])
    seed = int(settings.get("seed", 42))
    fractions = {name: float(settings[f"{name}_split_size"])
                 for name in ("train", "validation", "test")}
    if not all(0 < x < 1 for x in fractions.values()) or not np.isclose(sum(fractions.values()), 1):
        raise ValueError("All three split fractions must be positive and sum to one.")
    text_column = ds_config.get("text_column", "text")
    label_column = ds_config.get("label_column", "label")
    occurrence = Counter()
    records = {}
    groups = []
    for index, example in enumerate(dataset):
        text, label = example[text_column], int(example[label_column])
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty or invalid text at row {index}.")
        digest = _sha(_json([text, label]))
        example_id = f"{digest}:{occurrence[digest]}"
        occurrence[digest] += 1
        group_id = _sha(text.encode("utf-8"))
        records[example_id] = (index, group_id, label)
        groups.append(group_id)
    if len(set(groups)) < 3:
        raise ValueError("At least three distinct text groups are required.")
    signature = {
        "schema_version": 1,
        "dataset_sha256": _sha(_json(sorted(records))),
        "seed": seed, "fractions": fractions,
        "preprocessing_version": settings.get("preprocessing_version", "unspecified"),
        "method": "GroupShuffleSplit on exact prepared text; no stratification",
    }
    assignments_path = directory / "split_assignments.csv"
    metadata_path = directory / "metadata.json"
    if directory.exists():
        if not assignments_path.is_file() or not metadata_path.is_file():
            raise ValueError(f"Incomplete split directory: {directory}. Use a new version directory.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            raise ValueError("Dataset or split settings changed. Use a new split version; existing splits are never overwritten.")
        assignment_bytes = assignments_path.read_bytes()
        if _sha(assignment_bytes) != metadata["assignments_sha256"]:
            raise ValueError("Split assignment checksum mismatch; file was modified.")
    else:
        indices = np.arange(len(dataset))
        groups_array = np.asarray(groups)
        first = GroupShuffleSplit(n_splits=1, test_size=fractions["test"], random_state=seed)
        remaining, test = next(first.split(indices, groups=groups_array))
        second = GroupShuffleSplit(
            n_splits=1, test_size=fractions["validation"] / (fractions["train"] + fractions["validation"]),
            random_state=seed,
        )
        try:
            train_relative, val_relative = next(second.split(remaining, groups=groups_array[remaining]))
        except ValueError as exc:
            raise ValueError("Too few text groups for the requested split fractions.") from exc
        assignment = {}
        for name, selected in (("train", remaining[train_relative]),
                               ("validation", remaining[val_relative]), ("test", test)):
            assignment.update({int(index): name for index in selected})
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["example_id", "text_group_id", "split"])
        for example_id, (index, group_id, _) in sorted(records.items()):
            writer.writerow([example_id, group_id, assignment[index]])
        assignment_bytes = buffer.getvalue().encode("utf-8")
        local_source = ds_config.get("local", {}).get("path") if ds_config.get("source") == "local" else None
        group_labels = {}
        for _, group_id, label in records.values():
            group_labels.setdefault(group_id, set()).add(label)
        metadata = {
            "signature": signature,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "assignments_sha256": _sha(assignment_bytes),
            "source_path": local_source,
            "source_sha256": _sha(Path(local_source).read_bytes()) if local_source else None,
            "preprocessing_version": settings.get("preprocessing_version", "unspecified"),
            "duplicate_policy": "Keep all rows; exact prepared text stays in one split, even with conflicting labels.",
            "conflicting_text_groups": sum(len(labels) > 1 for labels in group_labels.values()),
            "row_counts": dict(Counter(assignment.values())),
            "class_counts": {name: dict(Counter(str(records[key][2]) for key in records
                                                if assignment[records[key][0]] == name)) for name in fractions},
        }
        directory.mkdir(parents=True, exist_ok=False)
        assignments_path.write_bytes(assignment_bytes)
        metadata_path.write_bytes(_json(metadata))

    rows = list(csv.DictReader(io.StringIO(assignment_bytes.decode("utf-8"))))
    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(records):
        raise ValueError("Saved split does not cover each example exactly once.")
    selected = {name: [] for name in fractions}
    seen_groups = {}
    for row in rows:
        index, group_id, _ = records[row["example_id"]]
        name = row["split"]
        if name not in selected or row["text_group_id"] != group_id:
            raise ValueError("Invalid saved split assignment.")
        if group_id in seen_groups and seen_groups[group_id] != name:
            raise ValueError("Identical text crosses split boundaries.")
        seen_groups[group_id] = name
        selected[name].append(index)
    if any(not values for values in selected.values()):
        raise ValueError("Every split must contain examples.")
    result = DatasetDict({name: dataset.select(indices) for name, indices in selected.items()})
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    reference = {
        "split_version": directory.name,
        "assignments_path_relative_to_run": Path(os.path.relpath(assignments_path.resolve(), run_dir.resolve())).as_posix(),
        "assignments_sha256": _sha(assignment_bytes),
        "metadata_path_relative_to_run": Path(os.path.relpath(metadata_path.resolve(), run_dir.resolve())).as_posix(),
        "metadata_sha256": _sha(metadata_path.read_bytes()),
        "dataset_sha256": signature["dataset_sha256"],
    }
    reference_path = run_dir / "split_reference.json"
    if reference_path.exists() and json.loads(reference_path.read_text(encoding="utf-8")) != reference:
        raise ValueError("This run already references another split. Create a new run.")
    reference_path.write_bytes(_json(reference))
    if copy_assignments:
        for filename, content in (("split_assignments.csv", assignment_bytes), ("split_metadata.json", metadata_path.read_bytes())):
            target = run_dir / filename
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f"Refusing to overwrite different split snapshot: {target}")
            target.write_bytes(content)
    return result
