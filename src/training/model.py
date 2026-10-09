from __future__ import annotations
from pathlib import Path


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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

def load_tokenizer(model_name: str):
    """Load a Hugging Face tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)


def load_model(
    model_name: str,
    label_names: dict[int, str],
):
    """Load a sequence classification model."""

    id2label = label_names
    label2id = {
        label: idx
        for idx, label in id2label.items()
    }

    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )


def prepare_lora(model, num_labels: int, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
    """Load a model and prepare it for LoRA fine-tuning."""


    # Define LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["query", "value"],  # Target attention layers
        modules_to_save=["classifier"],  # Save the classifier layer
    )

    # Wrap the model with LoRA
    model = get_peft_model(model, lora_config)
    return model


def load_fine_tuned_model(model_path: str, label_names: dict[int, str]):
    """Load a fine-tuned sequence classification model."""
    saved_config = PeftConfig.from_pretrained(model_path)
    model_path = saved_config.base_model_name_or_path
    if not model_path:
        raise ValueError("In adapter_config.json fehlt base_model_name_or_path.")


    return load_model(model_path, label_names)


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


#needs to be 0!
def check_max_sequence_length(tokenizer, tokenized_datasets):
    """Check the maximum sequence length for a given tokenizer."""
    max_length = tokenizer.model_max_length

    count = 0
    for i in range(len(tokenized_datasets['train'])):
        t = tokenized_datasets['train'][i]['input_ids']
    if len(t) > max_length:
        count += 1
        #remowing row with more than 512 tokens

        #print("Found a sequence longer than 512 tokens at index:", i, "with length:", len(t))
        #print(r"{}".format(tokenized_datasets['train'][i]))
        
    for i in range(len(tokenized_datasets['test'])):
        t = tokenized_datasets['test'][i]['input_ids']
        if len(t) > max_length:
            count += 1
    for i in range(len(tokenized_datasets['validation'])):
        t = tokenized_datasets['validation'][i]['input_ids']
        if len(t) > max_length:
            count += 1
    return count




def tokenize_dataset(dataset, tokenizer):
    """Tokenize the dataset using the provided tokenizer."""
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    return tokenized_datasets

def compute_metrics_with3(eval_pred):
    predictions, labels = eval_pred

    # apply softmax (3 classes)
    exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    probabilities = exp_preds / exp_preds.sum(axis=1, keepdims=True)

    # Ensure probabilities are float32 and labels are int32
    probabilities = probabilities.astype(np.float32)
    labels = labels.astype(np.int32)

    # MULTI-CLASS ROC AUC (3 classes)
    auc = np.round(
        roc_auc_score(
            y_true=labels,
            y_score=probabilities,
            multi_class="ovr"   # or "ovo"??
        ),
        3
    )

    # predicted class
    predicted_classes = np.argmax(probabilities, axis=1)

    # accuracy
    acc = np.round(
        accuracy_score(
            y_true=labels,
            y_pred=predicted_classes
        ),
        3
    )

    # f1 score
    f1 = np.round(
        f1_score(
            y_true=labels,
            y_pred=predicted_classes,
            average="weighted"
        ),
        3
    )

    return {
        "accuracy": acc,
        "roc_auc": auc,
        "f1_score": f1
    }

def compute_metrics(eval_pred):
    probs, labels = eval_pred

   

    # 3. Get predicted class labels (index of highest probability)
    predictions = np.argmax(probs, axis=1)
    
    # 4. Calculate metrics using Scikit-Learn
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "F1 Score": f1_score(labels, predictions)
    }

def compute_predict(eval_pred):
    predictions, labels = eval_pred
    
    # apply softmax (3 classes)
    #exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    #probabilities = exp_preds / exp_preds.sum(axis=1, keepdims=True)

    pred = np.argmax(predictions, axis=1)
    return pred

""" def load_fine_tuned_model(model_name: str, num_labels: int, model_path: str):
    #Load a fine-tuned model from a specified path.
    lora_model = PeftModel.from_pretrained(
        model, best_model_path, is_trainable=False,
    )
    lora_model.eval()
    print("Gespeichertes LoRA-Modell geladen:", best_model_path)
    return model
 """
