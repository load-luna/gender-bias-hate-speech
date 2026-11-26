
import numpy as np

from src.evaluation.bootstrap import compute_paired_bootstrap_confidence_interval


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None

def _difference(value1, value2):
    if value1 is None or value2 is None:
        return None
    return float(value1 - value2)


# Mean Direction of Change
#positiv when probability increased after swap
#negative when probability decreased after swap
def mean_change(original_probs, swapped_probs, axis=-1):
    return np.mean(swapped_probs - original_probs, axis=axis)

# Absolute difference between original and swapped predictions
def ctf_gap(original_probs, swapped_probs, axis=-1):
    return np.mean(np.abs(swapped_probs - original_probs), axis=axis)





def compute_counterfactual_score_metrics(paired_predictions, class_ids):
    results = {}

    for class_id in class_ids:
        results[class_id] = {
            "mean_change": compute_paired_bootstrap_confidence_interval(
                paired_predictions,
                label_id=class_id,
                statistic_func=mean_change,
            ),
            "ctf_gap": compute_paired_bootstrap_confidence_interval(
                paired_predictions,
                label_id=class_id,
                statistic_func=ctf_gap,
            ),
        }

    return results



def sub_counterfactual(paired_prediction,class_ids,label_names):
    num_classes = len(class_ids)
    num_gender_labels = 4

    n_samples = len(paired_prediction["id"])
    

    gender_label_flip = 0
    prediction_flip = 0
    new_error = 0
    fix_error = 0
    stays_error = 0

    pred_matrix = np.zeros(shape=(num_classes, num_classes, num_classes))
    
    gender_matrix = np.zeros(shape=(num_gender_labels, num_gender_labels))
    empty_score = {
        "estimate": None,
        "confidence_interval": (None, None),
    }
    score_metrics = {
        class_id: {
            "mean_change": empty_score.copy(),
            "ctf_gap": empty_score.copy(),
        }
        for class_id in class_ids
    }

    #empty return
    if n_samples == 0:
            return {
                "n_samples": 0,
                "gender_label_flip": 0,
                "prediction_flip": 0,
                "gender_label_rate": None,
                "prediction_flip_rate": None,
                "new_error": 0,
                "fix_error": 0,
                "stays_error": 0,
                "prediction_swap_matrix" : {
                    "class_ids": class_ids,
                    "label_names": label_names,
                    "transition_matrix": {
        
                        "counts": pred_matrix,
                    },
                },
                "gender_label_swap_matrix" : {
                    "class_ids": [0, 1, 2, 3],
                    "label_names": ["male", "female", "equal", "none"],
                    "transition_matrix": {
                        "rows": "original_gender_label",
                        "columns": "swapped_gender_label",
                        "counts": gender_matrix,
                    },
                },
                "fairness_metrics": score_metrics,
            }
            


    for p in paired_prediction.itertuples(index=False):


        pred_matrix[p.label_original, p.prediction_original, p.prediction_swapped] += 1
        gender_matrix[p.gender_label_original,p.gender_label_swapped] += 1

        if p.prediction_original != p.prediction_swapped:
            prediction_flip+= 1

            if p.prediction_original == p.label_original: new_error += 1
            elif p.prediction_swapped == p.label_original: fix_error += 1
            else: stays_error+= 1

        if p.gender_label_original != p.gender_label_swapped: gender_label_flip += 1



    p_rate = _ratio(prediction_flip , n_samples)
    g_rate = _ratio(gender_label_flip , n_samples)

    score_metrics = compute_counterfactual_score_metrics(
        paired_prediction,
        class_ids,
    )

    return {
        "n_samples" : n_samples,
        "gender_label_flip" : gender_label_flip,
        "prediction_flip" : prediction_flip,
        "gender_label_rate" : g_rate,
        "prediction_flip_rate" : p_rate,
        "new_error" : new_error,
        "fix_error" : fix_error,
        "stays_error" : stays_error,
        "prediction_swap_matrix" : {
            "class_ids": class_ids,
            "label_names": label_names,
            "transition_matrix": {

                "counts": pred_matrix,
            },
        },
        "gender_label_swap_matrix" : {
            "class_ids": [0, 1, 2, 3],
            "label_names": ["male", "female", "equal", "none"],
            "transition_matrix": {
                "rows": "original_gender_label",
                "columns": "swapped_gender_label",
                "counts": gender_matrix,
            },
        },
        "fairness_metrics": score_metrics,
    }



def eval_counterfactual(paired_prediction,class_ids,label_names):

    evaluation = {}

    mask_all_with_gender_terms = (paired_prediction["gender_label_original"] != 3)

    mask_male_to_female = (
        (paired_prediction["gender_label_original"] == 0)
        & (paired_prediction["gender_label_swapped"] == 1)
    )

    mask_female_to_male = (
        (paired_prediction["gender_label_original"] == 1)
        & (paired_prediction["gender_label_swapped"] == 0)
    )

    mask_equal_to_equal = (
        (paired_prediction["gender_label_original"] == 2)
        & (paired_prediction["gender_label_swapped"] == 2)
    )

    mask_other = ~(mask_male_to_female|mask_female_to_male|mask_equal_to_equal)


    all_with_gender_terms = paired_prediction[mask_all_with_gender_terms]
    male_to_female = paired_prediction[mask_male_to_female]
    female_to_male = paired_prediction[mask_female_to_male]
    equal_to_equal = paired_prediction[mask_equal_to_equal]
    other = paired_prediction[mask_other]




    evaluation["all"] = sub_counterfactual(paired_prediction,class_ids,label_names)
    evaluation["all_with_gender_terms"] = sub_counterfactual(all_with_gender_terms,class_ids,label_names)
    evaluation["male_to_female"] = sub_counterfactual(male_to_female,class_ids,label_names)
    evaluation["female_to_male"] = sub_counterfactual(female_to_male,class_ids,label_names)
    evaluation["equal_to_equal"] = sub_counterfactual(equal_to_equal,class_ids,label_names)
    evaluation["other"] = sub_counterfactual(other,class_ids,label_names)



    return evaluation
