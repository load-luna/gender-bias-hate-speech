# ============================================================
# 03_baseline.ipynb
# Baseline experiment
#
# BERT + LoRA
# No bias mitigation
# ============================================================


# ============================================================
# 1. SETUP
# ============================================================

# Only needed in Google Colab:
# %cd /content/gender-bias-hate-speech
# !pip install -r requirements.txt


# ============================================================
# 2. IMPORTS
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.manager import ExperimentConfig, prepare_run

from src.data.dataset import load_dataset
from src.data.text_processing import tokenize_dataset

from src.training.model import load_model, load_tokenizer
from src.training.train import (
    create_training_arguments,
    create_trainer,
)

from src.evaluation.metrics import classification_metrics
from src.evaluation.fairness import equalized_odds_gaps


# ============================================================
# 3. LOAD CONFIGURATION
# ============================================================

config = ExperimentConfig.from_files(
    model="configs/model.yaml",
    training="configs/training.yaml",
    evaluation="configs/evaluation.yaml",
    dataset="configs/datasets/dataset_a.yaml",
    mitigations=[
        "configs/mitigation/baseline.yaml",
    ],
)

print("Resolved configuration:")
print(config.to_dict())


# ============================================================
# 4. CREATE RUN
# ============================================================

config, run_dir = prepare_run(
    experiment_name="baseline_dataset_a",
    config=config,
)

print(f"Run directory: {run_dir}")


# ============================================================
# 5. LOAD DATASET
# ============================================================

dataset = load_dataset(
    config["dataset"]
)

print(dataset)


# ============================================================
# 6. DATASET SANITY CHECKS
# ============================================================

print("\nDataset splits:")

for split in dataset.keys():
    print(f"{split}: {len(dataset[split])}")


print("\nColumns:")
print(dataset["train"].column_names)


print("\nFirst training example:")
print(dataset["train"][0])


# Target label distribution
if "label" in dataset["train"].column_names:

    print("\nLabel distribution:")

    print(
        dataset["train"]
        .to_pandas()["label"]
        .value_counts()
        .sort_index()
    )


# Gender-label distribution
if "gender_label" in dataset["train"].column_names:

    print("\nGender distribution:")

    print(
        dataset["train"]
        .to_pandas()["gender_label"]
        .value_counts()
        .sort_index()
    )


# ============================================================
# 7. CHECK BASELINE MITIGATION CONFIG
# ============================================================

mitigation_config = config.get(
    "mitigation",
    {}
)

print("\nMitigation configuration:")
print(mitigation_config)


# Baseline should not have active mitigation.
for mitigation_name, settings in mitigation_config.items():

    if isinstance(settings, dict):

        if settings.get("enabled", False):

            raise ValueError(
                f"Baseline experiment has active mitigation: "
                f"{mitigation_name}"
            )


print("Baseline confirmed: no mitigation enabled.")


# ============================================================
# 8. LOAD TOKENIZER
# ============================================================

tokenizer = load_tokenizer(
    config["model"]
)

print("\nTokenizer loaded:")
print(tokenizer.__class__.__name__)


# ============================================================
# 9. LOAD MODEL
# ============================================================

# load_model() should:
#
# 1. load bert-base-uncased
# 2. create the sequence-classification head
# 3. apply LoRA
#
# LoRA is part of the standard model setup and therefore
# should happen inside src/training/model.py.

model = load_model(
    config["model"]
)

print("\nModel loaded:")
print(model.__class__.__name__)


# Check that LoRA is actually active.
if hasattr(model, "print_trainable_parameters"):

    print("\nTrainable parameters:")

    model.print_trainable_parameters()


# ============================================================
# 10. TOKENIZE DATASET
# ============================================================

tokenized_dataset = tokenize_dataset(
    dataset=dataset,
    tokenizer=tokenizer,
    config=config["dataset"],
)

print("\nTokenized dataset:")
print(tokenized_dataset)


# ============================================================
# 11. TRAINING ARGUMENTS
# ============================================================

training_args = create_training_arguments(
    config=config["training"],
    output_dir=run_dir / "checkpoints",
)

print("\nTraining arguments:")
print(training_args)


# ============================================================
# 12. CREATE TRAINER
# ============================================================

trainer = create_trainer(
    model=model,
    tokenizer=tokenizer,
    training_args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

print("\nTrainer created.")


# ============================================================
# 13. TRAIN MODEL
# ============================================================

train_result = trainer.train()


print("\nTraining finished.")

print("\nTraining metrics:")
print(train_result.metrics)


# Save Hugging Face training metrics
trainer.save_metrics(
    "train",
    train_result.metrics,
)


# ============================================================
# 14. TEST SET PREDICTIONS
# ============================================================

prediction_output = trainer.predict(
    tokenized_dataset["test"]
)


logits = prediction_output.predictions

y_true = prediction_output.label_ids

y_pred = np.argmax(
    logits,
    axis=1,
)


print("\nNumber of predictions:")
print(len(y_pred))


# ============================================================
# 15. PERFORMANCE METRICS
# ============================================================

performance = classification_metrics(
    y_true=y_true,
    y_pred=y_pred,
)

print("\nPerformance metrics:")

for name, value in performance.items():
    print(f"{name}: {value}")


# ============================================================
# 16. FAIRNESS METRICS
# ============================================================

fairness = {}


if "gender_label" in tokenized_dataset["test"].column_names:

    gender_labels = tokenized_dataset["test"]["gender_label"]

    fairness = equalized_odds_gaps(
        y_true=y_true,
        y_pred=y_pred,
        group_labels=gender_labels,
    )

    print("\nFairness metrics:")

    for name, value in fairness.items():
        print(f"{name}: {value}")

else:

    print(
        "\nNo gender_label column found. "
        "Fairness evaluation skipped."
    )


# ============================================================
# 17. SAVE INDIVIDUAL PREDICTIONS
# ============================================================

prediction_data = {
    "text": dataset["test"]["text"],
    "label": y_true,
    "prediction": y_pred,
}


if "gender_label" in dataset["test"].column_names:

    prediction_data["gender_label"] = (
        dataset["test"]["gender_label"]
    )


predictions_df = pd.DataFrame(
    prediction_data
)


predictions_path = (
    run_dir / "predictions.csv"
)


predictions_df.to_csv(
    predictions_path,
    index=False,
)


print(
    f"\nPredictions saved to: "
    f"{predictions_path}"
)


# ============================================================
# 18. SAVE METRICS
# ============================================================

metrics = {
    "performance": performance,
    "fairness": fairness,
}


metrics_path = (
    run_dir / "metrics.json"
)


with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=2,
    )


print(
    f"Metrics saved to: "
    f"{metrics_path}"
)


# ============================================================
# 19. OPTIONAL: SAVE FINAL MODEL
# ============================================================

model_dir = (
    run_dir / "model"
)


trainer.save_model(
    model_dir
)


tokenizer.save_pretrained(
    model_dir
)


print(
    f"Model saved to: "
    f"{model_dir}"
)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("BASELINE EXPERIMENT FINISHED")
print("=" * 60)

print(
    f"Run directory: "
    f"{run_dir}"
)

print("\nPerformance:")

for name, value in performance.items():

    print(
        f"  {name}: "
        f"{value:.4f}"
    )


if fairness:

    print("\nFairness:")

    for name, value in fairness.items():

        print(
            f"  {name}: "
            f"{value:.4f}"
        )


print("\nSaved files:")

print(
    f"  Config: "
    f"{run_dir / 'config.yaml'}"
)

print(
    f"  Metrics: "
    f"{metrics_path}"
)

print(
    f"  Predictions: "
    f"{predictions_path}"
)

print(
    f"  Model: "
    f"{model_dir}"
)

print("=" * 60)