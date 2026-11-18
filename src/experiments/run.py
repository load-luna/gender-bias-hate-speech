"""One fresh, reproducible training run, independent of notebook state."""
from copy import deepcopy
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset

from src.config import ExperimentConfig, prepare_run
from src.data.dataset import load_data_from_config, prepare_dataset
from src.data.text_processing import deep_clean_tweet
from src.data.gender_labeling import add_gender_label, swap_gender_terms_dataset, gender_labels
from src.data.splits import load_or_create_splits
from src.mitigation.apply import resolve_mitigation_config, apply_mitigations
from src.training.model import load_tokenizer, load_model, load_fine_tuned_model, prepare_lora, tokenize_dataset, compute_metrics
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
    if set(tokens) - {"max_length", "truncation"}:
        raise ValueError("tokenization supports max_length and truncation")
    tokens.setdefault("max_length", 512)
    tokens.setdefault("truncation", True)
    if type(tokens["max_length"]) is not int or tokens["max_length"] <= 0:
        raise ValueError("max_length must be a positive integer")
    if type(tokens["truncation"]) is not bool:
        raise ValueError("truncation must be boolean")
    if data["mitigation"]["clp"]["enabled"] and not tokens["truncation"]:
        raise ValueError("CLP requires truncation=true; use the same setting for comparisons")
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
        if evaluation["run_test"] or not evaluation["counterfactual_testing"] or not tokens["truncation"]:
            raise ValueError("Checkpoint search requires run_test=false, counterfactual_testing=true and truncation=true")
    if training.get("early_stopping", {}).get("enabled", False) and not training["load_best_model_at_end"]:
        raise ValueError("Early stopping requires load_best_model_at_end=true in this runner")
    return ExperimentConfig(data=data, sources=list(config.sources) if isinstance(config, ExperimentConfig) else [])


def _evaluate(trainer, dataset, config, run_dir, label_names):
    """Save original and optional swapped predictions using the same tokenization."""
    options = config["tokenization"]
    tokenized = tokenize_dataset(dataset, trainer.processing_class, **options)
    original = predict_dataset(tokenized, trainer)
    save_predictions(original, run_dir / "predictions.csv", label_names=label_names)
    def evaluate(predictions, directory):
        result = eval_prediction(predictions, sorted(label_names),
                                 [label_names[k] for k in sorted(label_names)], gender_labels)
        save_evaluation(result, directory)
        return result
    evaluation = evaluate(original, run_dir / "normal")
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
    tokenized = tokenize_dataset(validation_dataset, tokenizer, **config["tokenization"])
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

        #
        n_before = len(dataset)
        dataset = dataset.filter(lambda row: bool(row["text"].strip())).map(add_gender_label)

        #split data
        splits = load_or_create_splits(dataset, config, run_dir, copy_assignments=True)

        #tokenize
        tokenizer = load_tokenizer(config["model"]["path"])

        #apply preprocessing mitigation
        train, reports = apply_mitigations(splits["train"], config.to_dict(), tokenizer)
        _save_json(run_dir / "mitigation_report.json", reports)

        #data report
        # Save lineage without large token arrays.
        lineage = [c for c in ("id", "source_id", "root_id", "label", "gender_label", "clp_valid")
                   if c in train.column_names]
        train.select_columns(lineage).to_csv(str(run_dir / "training_rows.csv"), index=False)
        validation = tokenize_dataset(splits["validation"], tokenizer, **config["tokenization"])
        if not len(train) or not len(validation):
            raise ValueError("Tokenization left an empty training or validation split")
        _save_json(run_dir / "data_report.json", {
            "n_empty_removed": n_before - len(dataset),
            "split_sizes": {k: len(v) for k, v in splits.items()},
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
