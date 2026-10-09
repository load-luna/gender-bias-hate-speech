# Methodology Notes

Use this file to keep implementation decisions aligned with the written
methodology of the Bachelor thesis.

## Dataset and labels

Document:
- dataset source and version
- hate-speech target labels
- gender/group labels
- how gender labels were assigned
- handling of ambiguous / neutral cases

## Baseline

Document the exact BERT checkpoint, preprocessing, training split,
hyperparameters, random seed, and evaluation procedure.

## Mitigation

Document the exact implementation and rationale for:
- balancing
- gender swapping / augmentation
- fairness constraints

## Evaluation

Document:
- performance metrics
- fairness metrics
- subgroup definitions
- counterfactual testing procedure
