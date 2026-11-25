"""One fresh, reproducible training run, independent of notebook state."""
from copy import deepcopy
import gc
import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from datasets import Dataset

from src.config import ExperimentConfig, prepare_run
from src.data.dataset import load_data_from_config, prepare_dataset
from src.data.text_processing import deep_clean_tweet
from src.data.gender_labeling import (
    add_gender_label,
    gender_labels,
    gender_pair_side_presence,
    swap_gender_for_pairs,
    swap_gender_terms_dataset,
)
from src.data.splits import load_or_create_splits
from src.data.tokenization import (
    filter_pair_length_eligible,
    resolve_tokenization_options,
    tokenize_dataset,
)
from src.mitigation.apply import resolve_mitigation_config, apply_mitigations
from src.training.model import load_tokenizer, load_model, load_fine_tuned_model, prepare_lora, compute_metrics
from src.training.train import create_training_arguments, create_experiment_trainer, create_trainer, set_deterministic_seed
from src.training.predict import predict_dataset
from src.evaluation.metrics import eval_prediction
from src.evaluation.saving import save_predictions, save_counterfactual_predictions, save_evaluation, _json_values


def _save_json(path, value):
    Path(path).write_text(json.dumps(_json_values(value), indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")


def resolve_run_config(config):
    """Resolve defaults BEFORE the immutable run snapshot is written."""
    data = config.to_dict() if isinstance(config, ExperimentConfig) else deepcopy(config)
    data["mitigation"] = resolve_mitigation_config(data)
    tokens = data.setdefault("tokenization", {})
    if "truncation" in tokens:
        if "overlength_policy" in tokens:
            raise ValueError("Use tokenization.overlength_policy; do not also set truncation")
        legacy_truncation = tokens.pop("truncation")
        if type(legacy_truncation) is not bool:
            raise ValueError("Legacy tokenization.truncation must be boolean")
        tokens["overlength_policy"] = (
            "truncate" if legacy_truncation else "filter_pairs"
        )
    if set(tokens) - {"max_length", "overlength_policy"}:
        raise ValueError("tokenization supports max_length and overlength_policy")
    tokens.setdefault("max_length", 512)
    tokens.setdefault("overlength_policy", "filter_pairs")
    if type(tokens["max_length"]) is not int or tokens["max_length"] <= 0:
        raise ValueError("max_length must be a positive integer")
    if tokens["overlength_policy"] not in {"filter_pairs", "truncate"}:
        raise ValueError("overlength_policy must be 'filter_pairs' or 'truncate'")
    if data["model"].get("is_fine_tuned", False):
        raise ValueError("run_experiment trains fresh models. Use the analysis notebook for saved models.")
    training = data["training"]
    for key, value in {"fp16": False, "bf16": False, "gradient_accumulation_steps": 1,
                       "save_total_limit": 2}.items():
        training.setdefault(key, value)
    evaluation = data.setdefault("evaluation", {})
    evaluation.setdefault("batch_size", training["batch_size"])
    evaluation.setdefault("counterfactual_testing", True)
    evaluation.setdefault("run_test", True)
    evaluation.setdefault("validation_fairness", False)
    evaluation.setdefault("checkpoint_fairness", False)
    if evaluation["checkpoint_fairness"]:
        if (training.get("early_stopping", {}).get("enabled", False)
                or training["load_best_model_at_end"] or training["save_total_limit"] is not None):
            raise ValueError("Checkpoint search requires early stopping off, load_best_model_at_end=false and save_total_limit=null")

    if training.get("early_stopping", {}).get("enabled", False) and not training["load_best_model_at_end"]:
        raise ValueError("Early stopping requires load_best_model_at_end=true in this runner")
    return ExperimentConfig(data=data, sources=list(config.sources) if isinstance(config, ExperimentConfig) else [])


def _evaluate(trainer, dataset, config, run_dir, label_names):
    """Save original and optional swapped predictions using the same tokenization."""
    options = resolve_tokenization_options(config["tokenization"])
    tokenized = tokenize_dataset(dataset, trainer.processing_class, **options)
    original = predict_dataset(tokenized, trainer)
    save_predictions(original, run_dir / "predictions.csv", label_names=label_names)
    def evaluate(predictions, directory):
        result = eval_prediction(predictions, sorted(label_names),
                                 [label_names[k] for k in sorted(label_names)], gender_labels)
        save_evaluation(result, directory)
        return result
    evaluation = evaluate(original, run_dir / "normal")
    holdout = config["mitigation"].get("held_out_gender_terms", {})
    if holdout.get("enabled", False):
        for pair in holdout["pairs"]:
            male_term, female_term = pair
            selected = dataset.filter(
                lambda row: (
                    gender_pair_side_presence(row["text"], pair)[0]
                    != gender_pair_side_presence(row["text"], pair)[1]
                ),
                desc=f"Select held-out pair {male_term}/{female_term}",
            )
            if not len(selected):
                raise ValueError(f"No evaluation examples contain exactly one side of {pair}.")
            source_sides = [
                gender_pair_side_presence(text, pair)[0]
                for text in selected["text"]
            ]
            swapped = selected.map(
                lambda row: {
                    **row,
                    "text": swap_gender_for_pairs(row["text"], [pair]),
                },
                desc=f"Swap held-out pair {male_term}/{female_term}",
            ).map(add_gender_label)
            pair_tokens = tokenize_dataset(selected, trainer.processing_class, **options)
            swapped_tokens = tokenize_dataset(swapped, trainer.processing_class, **options)
            pair_original = predict_dataset(pair_tokens, trainer)
            pair_swapped = predict_dataset(swapped_tokens, trainer)
            pair_dir = run_dir / "held_out_gender_terms" / f"{male_term}__{female_term}"
            save_predictions(pair_original, pair_dir / "original_predictions.csv",
                             label_names=label_names,
                             metadata={"held_out_pair": pair, "source_side_only": True})
            save_predictions(pair_swapped, pair_dir / "swapped_predictions.csv",
                             label_names=label_names,
                             metadata={"held_out_pair": pair, "swap_scope": "selected pair only"})
            paired_path = save_counterfactual_predictions(
                pair_original, pair_swapped, pair_dir / "paired_predictions.csv",
                label_names=label_names,
                metadata={"held_out_pair": pair, "source_side_only": True},
            )
            delta = pair_swapped["probabilities"][:, 1] - pair_original["probabilities"][:, 1]
            labels = np.asarray(pair_original["label"])
            flips = np.asarray(pair_original["prediction"]) != np.asarray(pair_swapped["prediction"])

            def pair_summary(mask):
                if not mask.any():
                    return {"n": 0, "mean_change": None, "ctf_gap": None, "flip_rate": None}
                return {
                    "n": int(mask.sum()),
                    "mean_change": float(delta[mask].mean()),
                    "ctf_gap": float(np.abs(delta[mask]).mean()),
                    "flip_rate": float(flips[mask].mean()),
                }

            source_sides = np.asarray(source_sides, dtype=bool)
            summary = {
                "held_out_pair": pair,
                "n_pairs": len(selected),
                "probability_class_id": 1,
                "overall": pair_summary(np.ones(len(selected), dtype=bool)),
                "by_source_term": {
                    male_term: pair_summary(source_sides),
                    female_term: pair_summary(~source_sides),
                },
                "by_label": {
                    str(class_id): pair_summary(labels == class_id)
                    for class_id in sorted(set(labels.tolist()))
                },
                "paired_predictions": paired_path.name,
            }
            _save_json(pair_dir / "metrics.json", summary)
    if config["evaluation"]["counterfactual_testing"]:
        swapped = dataset.map(swap_gender_terms_dataset).map(add_gender_label)
        # Keep swap eligibility metadata for subsequent counterfactual analysis.
        columns = [c for c in ("id", "swap_grammar_uncertain", "swap_grammar_uncertain_terms")
                   if c in swapped.column_names]
        swapped.select_columns(columns).to_csv(str(run_dir / "swap_metadata.csv"), index=False)
        swap_tokens = tokenize_dataset(swapped, trainer.processing_class, **options)
        swap_predictions = predict_dataset(swap_tokens, trainer)
        save_predictions(swap_predictions, run_dir / "swapped_predictions.csv", label_names=label_names)
        save_counterfactual_predictions(original, swap_predictions,
                                        run_dir / "paired_predictions.csv", label_names=label_names)
        evaluate(swap_predictions, run_dir / "swapped")
    return evaluation


def evaluate_checkpoints(config, run_dir, validation_dataset, tokenizer, names):
    """Evaluate every saved epoch on validation; never accept a test dataset here.

    Runs after training so evaluation cannot change training RNG or early stopping.
    Model paths refer to actual full-model or LoRA checkpoints, not run_dir/model.
    """
    from src.experiments.clp_search import summarize_validation
    checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[-1]))
    if not checkpoints:
        raise ValueError("No checkpoints available for fairness selection")
    tokenized = tokenize_dataset(
        validation_dataset, tokenizer,
        **resolve_tokenization_options(config["tokenization"]),
    )
    rows = []
    for checkpoint in checkpoints:
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        directory = run_dir / "checkpoint_evaluations" / checkpoint.name
        directory.mkdir(parents=True, exist_ok=True)
        model = trainer = None
        try:
            model = load_fine_tuned_model(checkpoint, names, config.get("lora", {}).get("enabled", False))
            args = create_training_arguments(config, directory)
            trainer = create_trainer(model, tokenizer, args, tokenized, tokenized, compute_metrics)
            _save_json(directory / "validation_metrics.json", trainer.evaluate())
            _evaluate(trainer, validation_dataset, config, directory / "validation", names)
            row = {"epoch": state["epoch"], "global_step": state["global_step"],
                   "checkpoint": str(checkpoint.resolve()),
                   "evaluation_dir": str(directory.resolve()), **summarize_validation(directory)}
            rows.append(row)
            _save_json(run_dir / "checkpoint_metrics.json", rows)
            pd.DataFrame(rows).to_csv(run_dir / "checkpoint_metrics.csv", index=False)
        finally:
            if trainer is not None:
                trainer.accelerator.free_memory()
            del trainer, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def run_experiment(config, experiment_name, results_root="results"):
    """Train, save and evaluate one independent run; return only small summaries.

    Run from the project root (config paths are relative to it). No model or
    dataset is reused between runs. evaluation.run_test=false is useful while
    choosing hyperparameters on validation. Test predictions retain all groups;
    detailed counterfactual/bootstrap analysis remains in the analysis notebook.
    """
    config = resolve_run_config(config)
    config, run_dir = prepare_run(experiment_name, config, results_root=results_root)
    model = trainer = None
    try:
        _save_json(run_dir / "status.json", {"status": "running"})
        print(f"Starting {experiment_name}: {run_dir}")

        #set seed
        set_deterministic_seed(config["training"]["seed"])

        #load data
        raw = load_data_from_config(config)
        if isinstance(raw, pd.DataFrame):
            raw = Dataset.from_pandas(raw, preserve_index=False)

        #Prepare & clean Dataset
        dataset = prepare_dataset(raw, config)
        dataset = dataset.map(deep_clean_tweet)

        #add gender_label
        n_before = len(dataset)
        dataset = dataset.filter(lambda row: bool(row["text"].strip())).map(add_gender_label)

        #split data
        splits = load_or_create_splits(dataset, config, run_dir, copy_assignments=True)

        tokenizer = load_tokenizer(config["model"]["path"])
        tokenization_options = resolve_tokenization_options(config["tokenization"])
        length_policy = config["tokenization"]["overlength_policy"]

        if length_policy == "filter_pairs":
            splits, length_report = filter_pair_length_eligible(
                splits,
                tokenizer,
                max_length=config["tokenization"]["max_length"],
            )
        else:
            n_rows = sum(len(split) for split in splits.values())
            length_report = {
                "method": "truncate",
                "max_length": config["tokenization"]["max_length"],
                "n_before": n_rows,
                "n_after": n_rows,
                "n_removed": 0,
            }
        _save_json(run_dir / "length_handling_report.json", length_report)

        # Apply text-level mitigation first. Tokenize every resulting training
        # example, including CLP swaps, in one shared stage afterward.
        train_raw, reports = apply_mitigations(splits["train"], config.to_dict())

        # Save row lineage before token arrays are added.
        lineage = [c for c in ("id", "source_id", "root_id", "label", "gender_label",
                               "clp_pair_candidate") if c in train_raw.column_names]
        train_raw.select_columns(lineage).to_csv(str(run_dir / "training_rows.csv"), index=False)
        train = tokenize_dataset(train_raw, tokenizer, **tokenization_options)
        for report in reports:
            if report.get("stage") == "clp":
                report["max_length"] = config["tokenization"]["max_length"]
                report["truncation"] = tokenization_options["truncation"]
                report["n_pairs"] = int(sum(train["clp_valid"]))
                report["n_identical_tokens"] = report["n_pair_candidates"] - report["n_pairs"]
        _save_json(run_dir / "mitigation_report.json", reports)

        validation = tokenize_dataset(
            splits["validation"], tokenizer, **tokenization_options
        )
        if not len(train) or not len(validation):
            raise ValueError("Tokenization left an empty training or validation split")
        _save_json(run_dir / "data_report.json", {
            "n_empty_removed": n_before - len(dataset),
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "length_handling": length_report,
            "n_train_after_preparation": len(train), "n_validation_after_tokenization": len(validation),
        })

        # Reset immediately before initialization, independent of preprocessing RNG usage.
        set_deterministic_seed(config["training"]["seed"])

        #load Model set Trainer
        names = {int(k): v for k, v in config["dataset"]["label_mapping"]["names"].items()}
        model = load_model(config["model"]["path"], names)
        if config.get("lora", {}).get("enabled", False):
            model = prepare_lora(model, config)
        arguments = create_training_arguments(config, run_dir / "checkpoints")
        trainer = create_experiment_trainer(model, tokenizer, arguments, train, validation,
                                             config, compute_metrics)

        #train and save
        outcome = trainer.train()
        trainer.save_model(str(run_dir / "model"))
        tokenizer.save_pretrained(str(run_dir / "model"))
        trainer.save_state()
        validation_metrics = trainer.evaluate()
        _save_json(run_dir / "training_metrics.json", outcome.metrics)
        _save_json(run_dir / "validation_metrics.json", validation_metrics)

        #eval
        if config["evaluation"]["validation_fairness"]:
            _evaluate(trainer, splits["validation"], config, run_dir / "validation", names)
        test_metrics = None
        if config["evaluation"]["run_test"]:
            evaluation = _evaluate(trainer, splits["test"], config, run_dir, names)
            test_metrics = evaluation["groups"]["all"]["performance"]
        checkpoint_metrics = None
        if config["evaluation"]["checkpoint_fairness"]:
            trainer.accelerator.free_memory()
            trainer = model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            checkpoint_metrics = evaluate_checkpoints(config, run_dir, splits["validation"], tokenizer, names)
        result = {"experiment_name": experiment_name, "run_dir": str(run_dir),
                  "validation_metrics": validation_metrics, "test_metrics": test_metrics,
                  "checkpoint_metrics": checkpoint_metrics,
                  "mitigation_reports": reports}
        _save_json(run_dir / "summary.json", result)
        _save_json(run_dir / "status.json", {"status": "completed"})
        return _json_values(result)
    except Exception as exc:
        _save_json(run_dir / "status.json", {"status": "failed", "error": str(exc),
                                             "error_type": type(exc).__name__})
        raise
    finally:
        #get ready for next one
        if trainer is not None:
            trainer.accelerator.free_memory()
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
