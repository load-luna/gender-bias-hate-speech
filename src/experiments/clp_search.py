"""Summarize validation-only CLP searches from saved predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def summarize_validation(run_dir):
    """CTF on changed, grammar-safe male/female texts, probability of class 1.

    This population is defined by text, not model outputs or training lambda.
    Swaps truncated away may legitimately contribute zero to the metric.
    """
    run_dir = Path(run_dir)
    directory = run_dir / "validation"
    paired = pd.read_csv(directory / "paired_predictions.csv", dtype={"id": str})
    meta = pd.read_csv(directory / "swap_metadata.csv", dtype={"id": str})
    pairs = paired.merge(meta, on="id", validate="one_to_one", how="left", indicator=True)
    if not pairs["_merge"].eq("both").all():
        raise ValueError("Missing swap eligibility metadata")
    if not pairs["label_original"].eq(pairs["label_swapped"]).all():
        raise ValueError("Counterfactual labels do not align")
    uncertain = pairs["swap_grammar_uncertain"].astype(str).str.lower().map({"true": True, "false": False})
    if uncertain.isna().any():
        raise ValueError("Invalid grammar flags")
    eligible = (pairs["gender_label_original"].isin([0, 1]) & ~uncertain &
                pairs["text_original"].ne(pairs["text_swapped"]))
    pairs = pairs.loc[eligible].copy()
    if pairs.empty:
        raise ValueError("No eligible validation pairs; cannot select lambda using CTF gap")
    delta = pairs["probability_1_swapped"] - pairs["probability_1_original"]
    metrics = json.loads((run_dir / "validation_metrics.json").read_text())
    result = {"val_f1_macro": metrics["eval_f1_macro"], "val_accuracy": metrics["eval_accuracy"],
              "val_ctf_gap": float(delta.abs().mean()), "val_mean_change": float(delta.mean()),
              "val_flip_rate": float(pairs["prediction_original"].ne(pairs["prediction_swapped"]).mean()),
              "n_pairs": len(pairs)}
    for gender, direction in ((0, "male_to_female"), (1, "female_to_male")):
        subset = delta[pairs["gender_label_original"].eq(gender)]
        result[f"ctf_{direction}"] = float(subset.abs().mean()) if len(subset) else None
    # IDs allow the notebook to verify that every candidate used the same pairs.
    import hashlib
    result["pair_ids_sha256"] = hashlib.sha256(json.dumps(sorted(pairs["id"])).encode()).hexdigest()
    return result


def rank_candidates(rows, max_f1_drop=0.01):
    """Minimize mean CTF subject to mean F1 >= lambda=0 mean F1 - tolerance.

    All lambdas must have the exact same set of seeds, one run per seed.
    Standard deviations describe seeds; they are not confidence intervals.
    """
    frame = pd.DataFrame(rows)
    if not np.isfinite(max_f1_drop) or max_f1_drop < 0:
        raise ValueError("max_f1_drop must be finite and nonnegative")
    if frame.empty or frame.duplicated(["clp_lambda", "seed"]).any():
        raise ValueError("Require one result per lambda/seed")
    if not np.isfinite(frame[["val_f1_macro", "val_ctf_gap"]].to_numpy()).all():
        raise ValueError("Selection metrics must be finite")
    baseline = frame[frame["clp_lambda"].eq(0)]
    if baseline.empty:
        raise ValueError("Include lambda=0 as reference")
    expected = set(baseline["seed"])
    if any(set(group["seed"]) != expected for _, group in frame.groupby("clp_lambda")):
        raise ValueError("Every candidate must use the same seeds")
    if frame["pair_ids_sha256"].nunique() != 1:
        raise ValueError("Validation pair populations differ")
    table = frame.groupby("clp_lambda", as_index=False).agg(
        n_seeds=("seed", "count"), f1_mean=("val_f1_macro", "mean"), f1_std=("val_f1_macro", "std"),
        ctf_mean=("val_ctf_gap", "mean"), ctf_std=("val_ctf_gap", "std"),
        flip_mean=("val_flip_rate", "mean"), n_pairs=("n_pairs", "first"))
    threshold = baseline["val_f1_macro"].mean() - max_f1_drop
    table["f1_threshold"] = threshold
    table["eligible"] = table["f1_mean"] >= threshold
    table = table.sort_values(["eligible", "ctf_mean", "f1_mean", "clp_lambda"],
                               ascending=[False, True, False, True]).reset_index(drop=True)
    return table


def rank_checkpoint_candidates(rows, max_f1_drop=0.01):
    """Rank (lambda, epoch) across matched seeds using ONE baseline threshold.

    Baseline reference = maximum over epochs of mean seed F1 at lambda=0.
    Choose a common epoch across seeds, rather than cherry-picking each seed.
    """
    frame = pd.DataFrame(rows)
    if not np.isfinite(max_f1_drop) or max_f1_drop < 0:
        raise ValueError("max_f1_drop must be finite and nonnegative")
    keys = ["clp_lambda", "seed", "epoch"]
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError("Require exactly one result per lambda/seed/epoch")
    if not np.isfinite(frame[["val_f1_macro", "val_ctf_gap", "epoch"]].to_numpy()).all():
        raise ValueError("Checkpoint metrics must be finite")
    baseline = frame[frame["clp_lambda"].eq(0)]
    if baseline.empty:
        raise ValueError("Include lambda=0 checkpoints as reference")
    expected_seeds = set(baseline["seed"])
    expected_epochs = set(frame["epoch"])
    for _, group in frame.groupby(["clp_lambda", "epoch"]):
        if set(group["seed"]) != expected_seeds:
            raise ValueError("Every lambda/epoch must use the same seeds")
    for _, group in frame.groupby(["clp_lambda", "seed"]):
        if set(group["epoch"]) != expected_epochs:
            raise ValueError("Every run must include all epochs")
    if frame["pair_ids_sha256"].nunique() != 1:
        raise ValueError("Validation pair populations differ")
    table = frame.groupby(["clp_lambda", "epoch"], as_index=False).agg(
        n_seeds=("seed", "count"), f1_mean=("val_f1_macro", "mean"), f1_std=("val_f1_macro", "std"),
        ctf_mean=("val_ctf_gap", "mean"), ctf_std=("val_ctf_gap", "std"),
        flip_mean=("val_flip_rate", "mean"), n_pairs=("n_pairs", "first"))
    reference = table.loc[table["clp_lambda"].eq(0), "f1_mean"].max()
    table["baseline_f1_reference"] = reference
    table["f1_threshold"] = reference - max_f1_drop
    table["eligible"] = table["f1_mean"] >= table["f1_threshold"]
    return table.sort_values(["eligible", "ctf_mean", "f1_mean", "clp_lambda", "epoch"],
                             ascending=[False, True, False, True, True]).reset_index(drop=True)
