"""Independent binary CLP: original cross entropy plus squared logit-margin gap.

Prepare cleaned training data AFTER splitting and optional resampling/CDA. Each row keeps its
original ID/text/label; the counterfactual text is stored in an extra column, not as another row.
Validation and prediction use original cross entropy only. Lambda must be
selected on validation data. This is a margin-MSE variant, not an exact
reproduction of the original CLP paper's experimental setup.
"""

import math

import torch
from torch.nn import functional as F
from transformers import DataCollatorWithPadding, Trainer

from src.data.gender_labeling import (
    contains_gender_pair_term,
    get_grammar_uncertain_terms,
    swap_gender,
)


def prepare_clp_pairs(dataset, *, target_gender=(0, 1),
                      exclude_grammar_uncertain=True, class_ids=None,
                      exclude_gender_term_pairs=()):
    """Add CLP swap text and eligibility metadata without tokenizing.

    Token IDs for originals and swaps are created later by tokenize_dataset,
    so training examples share one tokenization stage.
    class_ids=None allows both labels; (0,) restricts pairing to one class
    without removing other classification rows.
    """
    required = {"text", "label", "gender_label"}
    if not required.issubset(dataset.column_names):
        raise ValueError(f"Required columns: {sorted(required)}")
    target_gender = tuple(target_gender)
    class_ids = None if class_ids is None else tuple(class_ids)

    def prepare(example):
        text = example["text"]
        selected = (example["gender_label"] in target_gender and
                    (class_ids is None or example["label"] in class_ids))
        held_out = selected and contains_gender_pair_term(text, exclude_gender_term_pairs)
        eligible = selected and not held_out
        uncertain = bool(get_grammar_uncertain_terms(text)) if eligible else False
        excluded = eligible and uncertain and exclude_grammar_uncertain
        pair_candidate = eligible and not excluded
        swapped = swap_gender(text) if pair_candidate else text
        return {
            "clp_text": swapped,
            "clp_selected": bool(selected),
            "clp_pair_candidate": bool(pair_candidate),
            "clp_excluded_grammar": bool(excluded),
            "clp_excluded_held_out": bool(held_out),
        }

    prepared = dataset.map(prepare, desc="Prepare CLP pair metadata")
    selected = sum(prepared["clp_selected"])
    excluded = sum(prepared["clp_excluded_grammar"])
    held_out_excluded = sum(prepared["clp_excluded_held_out"])
    candidates = sum(prepared["clp_pair_candidate"])
    report = {
        "method": "clp", "variant": "binary_margin_mse",
        "target_gender": list(target_gender),
        "class_ids": None if class_ids is None else list(class_ids),
        "exclude_grammar_uncertain": exclude_grammar_uncertain,
        "n_before": len(dataset), "n_after": len(prepared), "n_added": 0,
        "n_selected": int(selected), "n_excluded_grammar_uncertain": int(excluded),
        "n_excluded_held_out": int(held_out_excluded),
        "n_pair_candidates": int(candidates),
    }
    return prepared, report


class CLPDataCollator:
    """Pad originals and valid swaps separately; discard text/ID metadata.

    Also accepts ordinary tokenized validation/test rows without CLP columns.
    pair_indices links the smaller swap batch to rows in the original batch.
    """

    def __init__(self, tokenizer):
        self.input_names = tokenizer.model_input_names
        self.pad = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        originals, swaps, indices = [], [], []
        for index, feature in enumerate(features):
            original = {k: feature[k] for k in self.input_names if k in feature}
            if "labels" in feature or "label" in feature:
                original["labels"] = feature.get("labels", feature.get("label"))
            originals.append(original)
            if feature.get("clp_valid", False):
                indices.append(index)
                swaps.append({k: feature[f"clp_{k}"] for k in self.input_names
                              if f"clp_{k}" in feature})
        batch = self.pad(originals)
        batch["clp_pair_indices"] = torch.tensor(indices, dtype=torch.long)
        if swaps:
            batch.update({f"clp_{k}": v for k, v in self.pad(swaps).items()})
        return batch


def counterfactual_pairing_loss(original_logits, swapped_logits):
    """Mean squared difference of binary margins; gradients through BOTH sides."""
    if original_logits.shape != swapped_logits.shape:
        raise ValueError("Original and swapped logits must have equal shapes")
    if original_logits.ndim != 2 or original_logits.shape[1] != 2:
        raise ValueError("This CLP variant requires exactly two output logits")
    if original_logits.shape[0] == 0:
        return original_logits.sum() * 0.0
    original = original_logits.float()
    swapped = swapped_logits.float()
    return F.mse_loss(original[:, 1] - original[:, 0],
                      swapped[:, 1] - swapped[:, 0])


class CLPTrainer(Trainer):
    """Train with CE(originals) + clp_lambda * MSE(valid pair margins).

    Averages are per microbatch (all originals / valid pairs respectively).
    Use gradient_accumulation_steps=1 for the simplest interpretation; with
    accumulation the result averages microbatch losses, not all pairs globally.
    Standard training dropout remains active on both forward passes.
    """

    def __init__(self, *args, clp_lambda=1.0, compute_pairing_at_zero= False, **kwargs):
        if not math.isfinite(clp_lambda) or clp_lambda < 0:
            raise ValueError("clp_lambda must be finite and nonnegative")
        if kwargs.get("compute_loss_func") is not None:
            raise ValueError("CLPTrainer defines its own loss")
        super().__init__(*args, **kwargs)
        if self.args.remove_unused_columns:
            raise ValueError("Set training_args.remove_unused_columns = False for CLP")
        if not isinstance(self.data_collator, CLPDataCollator):
            raise ValueError("Pass data_collator=CLPDataCollator(tokenizer)")
        if self.args.label_smoothing_factor:
            raise ValueError("This CLP implementation uses plain cross entropy")
        self.clp_lambda = float(clp_lambda)
        # Trainer handles scaling for gradient accumulation; no item-count kwarg
        # is forwarded to BERT/PEFT, since we compute a mean loss ourselves.
        self.model_accepts_loss_kwargs = False
        #for loss diognostic
        self.compute_pairing_at_zero = compute_pairing_at_zero

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        indices = inputs.pop("clp_pair_indices", None)
        
        outputs = model(**inputs)
        logits = outputs.logits
        if logits.shape[-1] != 2:
            raise ValueError("CLPTrainer requires a binary two-logit classifier")
        if logits.shape != labels.shape:
            raise ValueError(f"logits: {logits.shape} must be equal to labels:{labels.shape}. Input: {input}")
        loss = F.cross_entropy(logits.float(), labels.long())

        if (
            model.training 
            and (self.clp_lambda > 0 or self.compute_pairing_at_zero)
            and indices is not None 
            and indices.numel()
        ):  
            
            swapped_inputs = {k[4:]: inputs.pop(k) for k in list(inputs)
                                      if k.startswith("clp_")}
            swapped_logits = model(**swapped_inputs).logits
            pairing = counterfactual_pairing_loss(logits[indices], swapped_logits)
            loss = loss + self.clp_lambda * pairing
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Also allow unlabeled prediction: no CLP keys reach model.forward().
        inputs = {k: v for k, v in inputs.items() if not k.startswith("clp_")}
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
