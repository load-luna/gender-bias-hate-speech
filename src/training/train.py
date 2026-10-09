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


    set_seed(training_config["seed"])

    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=training_config["learning_rate"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        num_train_epochs=training_config["epochs"],
        weight_decay=training_config["weight_decay"],
        warmup_ratio=training_config["warmup_ratio"],
        seed=training_config["seed"],
        data_seed=training_config["seed"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to="none",
    )


def create_trainer(
    model,
    tokenizer,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset=None,
    compute_metrics: Optional[Callable] = None,
) -> Trainer:
    """Create a standard Hugging Face Trainer."""
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )


def set_deterministic_seed(seed: int = 42) -> None:
    """Set common random seeds."""
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
