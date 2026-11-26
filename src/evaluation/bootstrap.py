from scipy.stats import bootstrap
import numpy as np

from src.data.gender_labeling import check_gender_label_in_dataset

def compute_paired_bootstrap_confidence_interval(paired_predictions, label_id, statistic_func, n_resamples=5000, confidence_level=0.95):
    """Compute paired bootstrap confidence interval for a given statistic."""
    if len(paired_predictions) == 0:
        raise ValueError("Cannot compute confidence interval for an empty dataset.")

    if not check_gender_label_in_dataset(paired_predictions,column_name="text_original"):
        return {"estimate": None, "confidence_interval": (None,None)}


    original_scores = paired_predictions[
        f"probability_{label_id}_original"
    ].to_numpy()

    swapped_scores = paired_predictions[
        f"probability_{label_id}_swapped"
    ].to_numpy()

    estimate = statistic_func(original_scores, swapped_scores)
    # scipy's bootstrap requires at least two paired observations. The point
    # estimate is still meaningful for a single pair, but its CI is not.
    if len(original_scores) < 2:
        return {"estimate": float(estimate), "confidence_interval": (None, None)}

    data = (original_scores, swapped_scores)

    result = bootstrap(
        data=data,
        statistic=statistic_func,
        paired=True,
        vectorized=True,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        method="BCa",
        batch=100,
        rng=np.random.default_rng(42),
    )

    ci = result.confidence_interval

    ci_low, ci_high = ci.low, ci.high
    if not (np.isfinite(ci_low) and np.isfinite(ci_high)):
        ci_low, ci_high = None, None

    return {"estimate": float(estimate), "confidence_interval": (ci_low, ci_high)}
