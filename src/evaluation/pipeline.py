from src.training.predict import predict_dataset
from src.evaluation.evaluate_classification import eval_prediction
from src.evaluation.saving import save_predictions, save_counterfactual_predictions, save_evaluation, save_json, save_counterfactual_eval
from src.evaluation.counterfactual import eval_counterfactual
from src.data.tokenization import (

    resolve_tokenization_options,
    tokenize_dataset,
)
from src.data.gender_labeling import (
    add_gender_label,
    gender_labels,
    swap_gender_terms_dataset,
    check_gender_label_in_dataset,
    contains_gender_pair_term,
    swap_gender_only_pairs,
    swap_only_pairs_dataset,
    add_gender_label_for_pair,
)

import numpy as np
import pandas as pd
from pathlib import Path




#tokenize,predict,evaluate,save
def _pipeline(trainer, dataset, config, run_dir, type):
    if not isinstance(type,str):
        raise ValueError("type needs to be String")
    options = resolve_tokenization_options(config["tokenization"])
    
    label_dict = {int(k): v for k, v in config["dataset"]["label_mapping"]["names"].items()}
    class_ids = sorted(label_dict)
    label_names = [label_dict[k] for k in sorted(label_dict)]

    tokenized = tokenize_dataset(dataset, trainer.processing_class, **options)
    prediction = predict_dataset(tokenized, trainer)
    save_predictions(
        prediction, 
        run_dir / f"{type}_predictions.csv", 
        label_names=label_dict
    )


    eval = eval_prediction(
        prediction,
        class_ids = class_ids, 
        label_names = label_names,
        gender_labels = gender_labels
    )

    save_evaluation(
        eval,
        run_dir / type,
        metadata={"variant": type, "model_path": str(config["model"]["path"]), "run_dir": str(run_dir)},
    )

    return {
        "prediction" : prediction,
        "evaluation" : eval,
    }

def evaluate(trainer, dataset, config, run_dir,*,test_pairs = ()):
    """Save original and optional swapped predictions using the same tokenization."""
    label_dict = {int(k): v for k, v in config["dataset"]["label_mapping"]["names"].items()}
    class_ids = sorted(label_dict)
    label_names = [label_dict[k] for k in sorted(label_dict)]

    

    original_results = _pipeline(trainer, dataset, config, run_dir, type="original")

    


    if config["evaluation"]["counterfactual_testing"]:
        if not check_gender_label_in_dataset(dataset):
            return
        
        swapped = dataset.map(swap_gender_terms_dataset).map(add_gender_label)
        # Keep swap eligibility metadata for subsequent counterfactual analysis.
        columns = [c for c in ("id", "swap_grammar_uncertain", "swap_grammar_uncertain_terms")
                    if c in swapped.column_names]
        swapped.select_columns(columns).to_csv(str(run_dir / "swap_metadata.csv"), index=False)

        swapped_results = _pipeline(trainer, swapped, config, run_dir, type="swapped")

        #save paired eval
        save_counterfactual_predictions(
            original_results["prediction"], 
            swapped_results["prediction"],
            run_dir / "paired_predictions.csv", 
            label_names=label_dict,
        )

        #need to be changed
        paired_scores = pd.read_csv(run_dir / "paired_predictions.csv", keep_default_na=False)
        counterfactual = eval_counterfactual(paired_scores,class_ids,label_names)

        save_counterfactual_eval(
            counterfactual,
            run_dir / "counterfactual",
            metadata={"selection": "all_pairs"},
        )


    
    

    #test pairs only
    if bool(test_pairs):
        
        for pair in test_pairs:
            pairs_only_dir = Path(run_dir / "pairs_only" / str(pair))
            
            pair_dataset = dataset.filter(
                lambda example: contains_gender_pair_term(example["text"], [pair])
            )

            if len(pair_dataset) == 0:
                # Paar überspringen
                continue

            #add gender_label only in terms of the pair
            pair_dataset = pair_dataset.map(
                add_gender_label_for_pair,
                fn_kwargs={"pair": pair},
            )
            
            original_pair_only_result = _pipeline(trainer, pair_dataset, config, pairs_only_dir, type="original")

            #test and swap held-out pairs only
            swapped_pair_dataset = pair_dataset.map(
                swap_only_pairs_dataset,
                fn_kwargs={"pairs": [pair]},
            )
            #add gender_label only in terms of the pair
            swapped_pair_dataset = swapped_pair_dataset.map(
                add_gender_label_for_pair,
                fn_kwargs={"pair": pair},
            )
            
            swapped_pair_only_result = _pipeline(trainer, swapped_pair_dataset, config, pairs_only_dir, type="swapped")

            #save paired eval
            save_counterfactual_predictions(
                original_pair_only_result["prediction"], 
                swapped_pair_only_result["prediction"],
                pairs_only_dir / "paired_predictions.csv", 
                label_names=label_dict,
                metadata={"selection": str(pair)},
            )

            #need to be changed
            pairs_only_paired_scores = pd.read_csv(pairs_only_dir / "paired_predictions.csv", keep_default_na=False)
            pairs_only_counterfactual = eval_counterfactual(pairs_only_paired_scores,class_ids,label_names)

            save_counterfactual_eval(
                pairs_only_counterfactual,
                pairs_only_dir / "counterfactual",
                metadata={"selection": str(pair)},
            )

    

