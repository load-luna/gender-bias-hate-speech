from __future__ import annotations
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, data
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np
from transformers import Trainer, TrainingArguments
from peft import (
    PeftModel,
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftConfig,
)
from src.training.predict import softmax


def load_tokenizer(model_path: str):
    """Load a Hugging Face tokenizer."""
    return AutoTokenizer.from_pretrained(model_path)


def load_model(
    model_path: str,
    label_names: dict[int, str],
):
    """Load a sequence classification model."""

    id2label = label_names
    label2id = {
        label: idx
        for idx, label in id2label.items()
    }

    return AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )


def prepare_lora(model, config):
    """""Load a model and prepare it for LoRA fine-tuning."""

    lora_config_values = config["lora"]
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=lora_config_values["r"],
        lora_alpha=lora_config_values["alpha"],
        lora_dropout=lora_config_values["dropout"],
        bias=lora_config_values["bias"],
        task_type=lora_config_values["task_type"],
        target_modules=lora_config_values["target_modules"],  # Target attention layers
        modules_to_save=lora_config_values["modules_to_save"],  # Save the classifier layer
    )

    # Wrap the model with LoRA
    model = get_peft_model(model, lora_config)
    return model


def load_fine_tuned_model(
    model_path: str | Path,
    label_names: dict[int, str],
    lora_enabled: bool = True,
):
    """Load a saved full model or LoRA adapter for inference."""
    model_path = str(model_path)

    # Full fine-tuning: load the saved model directly.
    if not lora_enabled:
        model = load_model(model_path, label_names)
        model.eval()
        return model
    
    # 1. Read the adapter configuration
    saved_config = PeftConfig.from_pretrained(model_path)
    base_model_path = saved_config.base_model_name_or_path

    if not base_model_path:
        raise ValueError(
            "In adapter_config.json fehlt base_model_name_or_path."
        )

    # 2. Load the original base model with the correct classification head
    base_model = load_model(
        model_path=base_model_path,
        label_names=label_names,
    )
    
    # 3. Load the trained adapter onto the base model
    lora_model = PeftModel.from_pretrained(
        base_model,
        model_path,
        is_trainable=False,
    )

    lora_model.eval()
    return lora_model

def freeze_model_parameters(model):
    """Freeze all parameters of the model except for the classifier layer."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_counts = {
        "Total": total_params,
        "Trainable": trainable_params,
        "Frozen": total_params - trainable_params
    }
    print("Model Parameter Counts:")
    for k, v in param_counts.items():
        print(f"{k}: {v:,}")


    return param_counts


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    # Manche Modelle liefern zusätzliche Outputs als Tupel.
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)

    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError(
            "Erwartet werden Logits mit Shape (n_samples, n_classes), "
            "mit mindestens zwei Klassen."
        )

    num_classes = logits.shape[1]
    class_ids = np.arange(num_classes)

    probabilities = softmax(logits)

    # Softmax verändert die Reihenfolge der Logits nicht.
    predicted_classes = np.argmax(logits, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(labels, predicted_classes)),
        "f1_score": float(
            f1_score(
                labels,
                predicted_classes,
                labels=class_ids,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                labels,
                predicted_classes,
                labels=class_ids,
                average="macro",
                zero_division=0,
            )
        ),
    }

    # Fehlt eine Klasse, ist die vollständige OvR-AUC nicht definiert.
    if not np.array_equal(np.unique(labels), class_ids):
        metrics["roc_auc"] = float("nan")
    elif num_classes == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(labels, probabilities[:, 1])
        )
    else:
        metrics["roc_auc"] = float(
            roc_auc_score(
                labels,
                probabilities,
                labels=class_ids,
                multi_class="ovr",
                average="macro",
            )
        )

    return metrics


#temporary Weighted Loss Function
def create_weighted_loss_function(config,tokenized_dataset):
    import torch
    import torch.nn.functional as F

    # Nur Labels aus dem Trainingsset verwenden
    train_labels = torch.tensor(
        tokenized_dataset["train"]["label"],
        dtype=torch.long,
    )

    num_classes = config["dataset"]["label_mapping"]["num_labels"]
    counts = torch.bincount(train_labels, minlength=num_classes)

    if (counts == 0).any():
        raise ValueError("Im Trainingsset fehlt mindestens eine Klasse.")

    class_weights = counts.sum() / (num_classes * counts.float())
    print(class_weights)


    def weighted_loss(outputs, labels, num_items_in_batch=None):
        logits = outputs.logits.float()
        weights = class_weights.to(logits.device)

        loss_sum = F.cross_entropy(
            logits,
            labels,
            weight=weights,
            reduction="sum",
        )

        denominator = (
            num_items_in_batch
            if num_items_in_batch is not None
            else labels.numel()
        )

        return loss_sum / denominator
