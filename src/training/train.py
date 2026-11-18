from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from datasets import config
import numpy as np
import torch

from src import config
from transformers import TrainingArguments, Trainer, set_seed


def create_training_arguments(
    config: dict,
    output_dir: str | Path,
    
) -> TrainingArguments:

    training_config = config["training"]


    

    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=training_config["learning_rate"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=config.get("evaluation", {}).get("batch_size", training_config["batch_size"]),
        num_train_epochs=training_config["epochs"],
        weight_decay=training_config["weight_decay"],
        warmup_steps=training_config["warmup_steps"],
        seed=training_config["seed"],
        data_seed=training_config["seed"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=training_config["greater_is_better"],
        logging_strategy="epoch",
        report_to="none",
        fp16=training_config.get("fp16", False),
        bf16=training_config.get("bf16", False),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        save_total_limit=training_config.get("save_total_limit"),
    )


def create_trainer(
    model,
    tokenizer,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset=None,
    compute_metrics: Optional[Callable] = None,
    compute_loss_func = None,
) -> Trainer:
    """Create a standard Hugging Face Trainer."""
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        compute_loss_func=compute_loss_func,
    )


def set_deterministic_seed(seed: int = 42) -> None:
    """Set common random seeds."""
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_experiment_trainer(model, tokenizer, training_args, train_dataset,
                              eval_dataset, config, compute_metrics=None):
    """Choose standard or CLP training and configure early stopping."""
    from transformers import EarlyStoppingCallback
    from src.mitigation.clp import CLPTrainer, CLPDataCollator

    clp = config["mitigation"]["clp"]
    if clp["enabled"]:
        training_args.remove_unused_columns = False
        trainer = CLPTrainer(
            model=model, args=training_args, processing_class=tokenizer,
            data_collator=CLPDataCollator(tokenizer), train_dataset=train_dataset,
            eval_dataset=eval_dataset, compute_metrics=compute_metrics,
            clp_lambda=clp["clp_lambda"],
        )
    else:
        trainer = create_trainer(model, tokenizer, training_args, train_dataset,
                                 eval_dataset, compute_metrics)
    early = config["training"].get("early_stopping", {})
    if early.get("enabled", False):
        trainer.add_callback(EarlyStoppingCallback(
            early_stopping_patience=early.get("patience", 2),
            early_stopping_threshold=early.get("threshold", 0.0),
        ))
    return trainer
