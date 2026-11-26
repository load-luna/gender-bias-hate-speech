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
from src.data.data_analysis import get_data_report
from src.data.text_processing import deep_clean_tweet
from src.data.gender_labeling import add_gender_label
from src.data.splits import load_or_create_splits
from src.data.tokenization import (
    filter_pair_length_eligible,
    resolve_tokenization_options,
    tokenize_dataset,
)
from src.mitigation.apply import resolve_mitigation_config, apply_mitigations
from src.training.model import load_tokenizer, load_model, load_fine_tuned_model, prepare_lora, compute_metrics
from src.training.train import create_training_arguments, create_experiment_trainer, create_trainer, set_deterministic_seed

from src.evaluation.saving import save_json, _json_values
from src.evaluation.pipeline import evaluate



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





def evaluate_checkpoints(config, run_dir, validation_dataset, tokenizer):
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
            label_dict = {int(k): v for k, v in config["dataset"]["label_mapping"]["names"].items()}
            model = load_fine_tuned_model(checkpoint, label_dict, config.get("lora", {}).get("enabled", False))
            args = create_training_arguments(config, directory)
            trainer = create_trainer(model, tokenizer, args, tokenized, tokenized, compute_metrics)
            save_json(directory / "validation_metrics.json", trainer.evaluate())
            evaluate(trainer, validation_dataset, config, directory / "validation")
            row = {"epoch": state["epoch"], "global_step": state["global_step"],
                   "checkpoint": str(checkpoint.resolve()),
                   "evaluation_dir": str(directory.resolve()), **summarize_validation(directory)}
            rows.append(row)
            save_json(run_dir / "checkpoint_metrics.json", rows)
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
        save_json(run_dir / "status.json", {"status": "running"})
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

        #filter for max_length even counterfactual
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
        save_json(run_dir / "length_handling_report.json", length_report)

        #check held-out terms
        held_out_settings = config["mitigation"]["held_out_gender_terms"]
        held_out_pairs = (
            held_out_settings["pairs"] if held_out_settings["enabled"] else []
        )
        #save data report
        dataset_analysis = get_data_report(
            splits, config.to_dict(), held_out_pairs=held_out_pairs
        )
        dataset_analysis["gender_class_distribution"].to_csv(
            run_dir / "gender_class_distribution.csv", index=False
        )
        if held_out_pairs:
            dataset_analysis["held_out_distribution"].to_csv(
                run_dir / "held_out_distribution.csv", index=False
            )

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
        save_json(run_dir / "mitigation_report.json", reports)

        validation = tokenize_dataset(
            splits["validation"], tokenizer, **tokenization_options
        )
        if not len(train) or not len(validation):
            raise ValueError("Tokenization left an empty training or validation split")
        save_json(run_dir / "data_report.json", {
            "n_empty_removed": n_before - len(dataset),
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "length_handling": length_report,
            "dataset_analysis": {
                "held_out_pairs": dataset_analysis["held_out_pairs"],
                "gender_class_distribution_file": "gender_class_distribution.csv",
                "held_out_distribution_file": (
                    "held_out_distribution.csv" if held_out_pairs else None
                ),
            },
            "n_train_after_preparation": len(train), "n_validation_after_tokenization": len(validation),
        })

        # Reset immediately before initialization, independent of preprocessing RNG usage.
        set_deterministic_seed(config["training"]["seed"])

        #load Model set Trainer
        label_dict = {int(k): v for k, v in config["dataset"]["label_mapping"]["names"].items()}
        model = load_model(config["model"]["path"], label_dict)
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
        save_json(run_dir / "training_metrics.json", outcome.metrics)
        save_json(run_dir / "validation_metrics.json", validation_metrics)

        #eval
        if config["evaluation"]["validation_fairness"]:
            evaluate(trainer, splits["validation"], config, run_dir / "validation")
        test_metrics = None
        if config["evaluation"]["run_test"]:
            test_pairs = config["evaluation"].get("test_pairs",())
            evaluate(trainer, splits["test"], config, run_dir, test_pairs = test_pairs)
        checkpoint_metrics = None
        if config["evaluation"]["checkpoint_fairness"]:
            trainer.accelerator.free_memory()
            trainer = model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            checkpoint_metrics = evaluate_checkpoints(config, run_dir, splits["validation"], tokenizer)

        #save run results
        result = {"experiment_name": experiment_name, "run_dir": str(run_dir),
                  "validation_metrics": validation_metrics,
                  "checkpoint_metrics": checkpoint_metrics,
                  "mitigation_reports": reports}
        save_json(run_dir / "summary.json", result)
        save_json(run_dir / "status.json", {"status": "completed"})
        return _json_values(result)
    except Exception as exc:
        save_json(run_dir / "status.json", {"status": "failed", "error": str(exc),
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
