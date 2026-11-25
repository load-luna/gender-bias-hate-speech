"""Prepare training data in the fixed order resampling -> CDA -> CLP."""
from copy import deepcopy
import math
from datasets import Dataset
from src.mitigation.resample import balance_gender_within_class
from src.mitigation.cda import selective_cda
from src.mitigation.clp import prepare_clp_dataset
from src.training.model import tokenize_dataset


def resolve_mitigation_config(config):
    """Resolve defaults and reject unknown options before running an experiment."""
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    for legacy in ("balancing", "gender_augmentation", "fairness_constraints"):
        if legacy in config:
            raise ValueError(f"Legacy key {legacy!r}: use mitigation.resampling/cda/clp")
    result = {
        "resampling": {"enabled": False, 
                       "strategy": "mean", 
                       "target_gender": [0, 1],
                       "class_id": sorted(int(k) for k in config["dataset"]["label_mapping"]["names"]),
                       "seed": config["training"]["seed"]},
        "cda": {"enabled": False, 
                "target_gender": [0, 1], 
                "exclude_grammar_uncertain": True,
                "seed": config["training"]["seed"]},
        "clp": {"enabled": False, 
                "clp_lambda": 0.1, 
                "target_gender": [0, 1],
                "exclude_grammar_uncertain": True, 
                "class_ids": None,
                "compute_pairing_at_zero": False,},
    }
    supplied = config.get("mitigation", {})
    if not isinstance(supplied, dict) or set(supplied) - set(result):
        raise ValueError("mitigation supports only resampling, cda and clp")
    classes = result["resampling"]["class_id"][:]
    for name, values in supplied.items():
        if not isinstance(values, dict) or set(values) - set(result[name]):
            raise ValueError(f"Unknown parameters for mitigation.{name}")
        result[name].update(deepcopy(values))
    for name, values in result.items():
        if type(values["enabled"]) is not bool:
            raise ValueError(f"mitigation.{name}.enabled must be boolean")
        genders = values["target_gender"]
        if not isinstance(genders, (list, tuple)) or not genders or len(set(genders)) != len(genders) or not set(genders) <= {0, 1, 2, 3}:
            raise ValueError(f"Invalid target_gender for {name}")
        if "exclude_grammar_uncertain" in values and type(values["exclude_grammar_uncertain"]) is not bool:
            raise ValueError("exclude_grammar_uncertain must be boolean")
    if result["resampling"]["strategy"] not in {"mean", "under", "over"}:
        raise ValueError("Resampling strategy must be mean, under or over")
    if sorted(result["resampling"]["class_id"]) != classes:
        raise ValueError("resampling.class_id must contain every class exactly once")
    clp = result["clp"]
    if not math.isfinite(clp["clp_lambda"]) or clp["clp_lambda"] < 0:
        raise ValueError("clp_lambda must be finite and nonnegative")
    if clp["class_ids"] is not None and (not clp["class_ids"] or not set(clp["class_ids"]) <= set(classes)):
        raise ValueError("clp.class_ids must select existing classes")
    if clp["enabled"] and classes != [0, 1]:
        raise ValueError("CLP requires binary labels 0 and 1")
    return result


def apply_mitigations(train_dataset, config, tokenizer):
    """Return tokenized train Dataset and ordered reports, without mutating input.

    CLP pairs ALL eligible rows after resampling/CDA, including CDA copies.
    source_id identifies the immediate parent; root_id identifies the original
    row before mitigation. Pass only the training split here.
    """
    settings = resolve_mitigation_config(config)
    tokens = config.get("tokenization", {})
    max_length = tokens.get("max_length", 512)
    truncation = tokens.get("truncation", True)
    if settings["clp"]["enabled"] and not truncation:
        raise ValueError("CLP requires tokenization.truncation=true")
    if not len(train_dataset):
        raise ValueError("Training dataset is empty")
    train, reports = train_dataset, []
    if settings["resampling"]["enabled"] or settings["cda"]["enabled"]:
        frame = train.to_pandas().copy()
        frame["id"] = frame["id"].astype("string")
        if not frame["id"].is_unique:
            raise ValueError("Input training IDs must be unique")
        if "root_id" not in frame:
            frame["root_id"] = frame["id"]
        for name, function in (("resampling", balance_gender_within_class), ("cda", selective_cda)):
            options = dict(settings[name])
            if options.pop("enabled"):
                frame, report = function(frame, **options)
                report["stage"] = name
                reports.append(report)
        train = Dataset.from_pandas(frame, preserve_index=False)
    if settings["clp"]["enabled"]:
        options = dict(settings["clp"])
        options.pop("enabled")
        weight = options.pop("clp_lambda")
        train, report = prepare_clp_dataset(train, tokenizer, max_length=max_length, **options)
        report.update(stage="clp", clp_lambda=weight, pair_scope="all_eligible_rows_after_preprocessing")
        reports.append(report)
    else:
        train = tokenize_dataset(train, tokenizer, truncation=truncation, max_length=max_length)
    return train, reports
