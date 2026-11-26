"""Prepare training data in the fixed order resampling -> CDA -> CLP."""
from copy import deepcopy
import math
from datasets import Dataset
from src.mitigation.resample import balance_gender_within_class
from src.mitigation.cda import selective_cda
from src.mitigation.clp import prepare_clp_pairs
from src.data.gender_labeling import normalize_gender_pairs


def resolve_mitigation_config(config):
    """Resolve defaults and reject unknown options before running an experiment."""
    if hasattr(config, "to_dict"):
        config = config.to_dict()

    dataset_class_ids = sorted(int(k) for k in config["dataset"]["label_mapping"]["names"])

    result = {
        "resampling": {"enabled": False, 
                       "strategy": "mean", 
                       "target_gender": [0, 1],
                       # Classes over which the resampling method iterates.
                       "class_id": dataset_class_ids[:],
                       "seed": config["training"]["seed"]},
        "cda": {"enabled": False, 
                "target_gender": [0, 1], 
                "exclude_grammar_uncertain": True,
                "seed": config["training"]["seed"]},
        "clp": {"enabled": False, 
                "clp_lambda": 0.1, 
                "target_gender": [0, 1],
                "exclude_grammar_uncertain": True, 
                # class_id = None means all classes
                "class_ids": None,
                "compute_pairing_at_zero": False,},
        "held_out_gender_terms": {
            "enabled": False, 
            "pairs": []},
    }
    #set config
    supplied = config.get("mitigation", {})
    if not isinstance(supplied, dict) or set(supplied) - set(result):
        raise ValueError("Unknown mitigation configuration section")
    
    for name, values in supplied.items():
        if not isinstance(values, dict) or set(values) - set(result[name]):
            raise ValueError(f"Unknown parameters for mitigation.{name}")
        result[name].update(deepcopy(values))

    #check mitigation settings
    for name, values in result.items():
        if name == "held_out_gender_terms":
            continue
        if type(values["enabled"]) is not bool:
            raise ValueError(f"mitigation.{name}.enabled must be boolean")
        genders = values["target_gender"]
        if not isinstance(genders, (list, tuple)) or not genders or len(set(genders)) != len(genders) or not set(genders) <= {0, 1, 2, 3}:
            raise ValueError(f"Invalid target_gender for {name}")
        if "exclude_grammar_uncertain" in values and type(values["exclude_grammar_uncertain"]) is not bool:
            raise ValueError("exclude_grammar_uncertain must be boolean")
        if "compute_pairing_at_zero" in values and type(values["compute_pairing_at_zero"]) is not bool:
            raise ValueError("mitigation.clp.compute_pairing_at_zero must be boolean")

    #held out terms
    held_out = result["held_out_gender_terms"]
    if type(held_out["enabled"]) is not bool:
        raise ValueError("mitigation.held_out_gender_terms.enabled must be boolean")
    if not isinstance(held_out["pairs"], (list, tuple)):
        raise ValueError("mitigation.held_out_gender_terms.pairs must be a list")
    normalized_pairs = normalize_gender_pairs(held_out["pairs"])
    if len(normalized_pairs) != len(held_out["pairs"]):
        raise ValueError("Held-out gender pairs must be unique.")
    if held_out["enabled"] and not normalized_pairs:
        raise ValueError("Provide at least one pair when held-out terms are enabled.")
    held_out["pairs"] = [list(pair) for pair in normalized_pairs]
    #resampling
    if result["resampling"]["strategy"] not in {"mean", "under", "over"}:
        raise ValueError("Resampling strategy must be mean, under or over")
    if sorted(result["resampling"]["class_id"]) != dataset_class_ids:
        raise ValueError("resampling.class_id must contain every dataset class exactly once")
    #clp
    clp = result["clp"]
    if not math.isfinite(clp["clp_lambda"]) or clp["clp_lambda"] < 0:
        raise ValueError("clp_lambda must be finite and nonnegative")

    if clp["class_ids"] is not None and (
        not clp["class_ids"] or not set(clp["class_ids"]) <= set(dataset_class_ids)
    ):
        raise ValueError("clp.class_ids must select one or more existing dataset classes")
    
    if clp["enabled"] and dataset_class_ids != [0, 1]:
        raise ValueError("CLP requires binary labels 0 and 1")
    return result


def apply_mitigations(train_dataset, config):
    """Apply text-level mitigation and return raw data plus ordered reports.

    CLP pairs ALL eligible rows after resampling/CDA, including CDA copies.
    source_id identifies the immediate parent; root_id identifies the original
    row before mitigation. Pass only the training split here.
    """
    settings = resolve_mitigation_config(config)
    if not len(train_dataset):
        raise ValueError("Training dataset is empty")
    train, reports = train_dataset, []
    held_out_pairs = (settings["held_out_gender_terms"]["pairs"]
                      if settings["held_out_gender_terms"]["enabled"] else [])
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
                if name == "cda":
                    options["held_out_pairs"] = held_out_pairs
                frame, report = function(frame, **options)
                report["stage"] = name
                reports.append(report)
        train = Dataset.from_pandas(frame, preserve_index=False)
    if settings["clp"]["enabled"]:
        options = dict(settings["clp"])
        options.pop("enabled")
        weight = options.pop("clp_lambda")
        options.pop("compute_pairing_at_zero", None)
        options["held_out_pairs"] = held_out_pairs
        train, report = prepare_clp_pairs(train, **options)
        report.update(stage="clp", clp_lambda=weight, pair_scope="all_eligible_rows_after_preprocessing")
        reports.append(report)
    return train, reports
