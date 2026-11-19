"""Build experiment configurations without loading models or starting training."""

from copy import deepcopy
from itertools import combinations, product
from pathlib import Path

from .manager import ExperimentConfig


def _set_value(values, path, value):
    parts = path.split(".")
    target = values
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            raise ValueError(f"Unknown config path: {path}")
        target = target[part]
    if parts[-1] not in target:
        raise ValueError(f"Unknown config path: {path}")
    target[parts[-1]] = deepcopy(value)


def build_experiment_plan(
    *, name, config_files, methods, combine_methods=False,
    include_baseline=True, seeds=(42,), run_test=False,
    overrides=None, hyperparameters=None, method_hyperparameters=None,
    mitigation_dir="configs/mitigation",
):
    """Return (name, config) pairs compatible with run_experiment.

    Grids use dotted config paths and lists of candidate values. Common grids
    apply to every variant; method grids only apply when that method is active.
    A method grid replaces the common grid for the same path. Conflicting grids
    from two active methods are rejected. Explicit overrides precede grids;
    seeds and run_test are controlled exclusively by their named arguments.
    """
    if isinstance(methods, str) or any(not isinstance(m, str) for m in methods):
        raise ValueError('methods must be a list of names, e.g. ["cda", "clp"]')
    if len(set(methods)) != len(methods) or "baseline" in methods:
        raise ValueError("Use unique method names and include_baseline for baseline")
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least one seed, without duplicates")
    overrides = overrides or {}
    hyperparameters = hyperparameters or {}
    method_hyperparameters = method_hyperparameters or {}
    reserved = {"training.seed", "evaluation.run_test"}
    for grid in [hyperparameters, *method_hyperparameters.values()]:
        for path, candidates in grid.items():
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"{path}: provide a non-empty list of candidates")
            if any(value == previous for i, value in enumerate(candidates)
                   for previous in candidates[:i]):
                raise ValueError(f"{path}: duplicate candidates")
    for mapping in [overrides, hyperparameters, *method_hyperparameters.values()]:
        if reserved.intersection(mapping):
            raise ValueError("Set seed and run_test using seeds and run_test")

    variants = [()] if include_baseline else []
    sizes = range(1, len(methods) + 1) if combine_methods else [1]
    variants += [combo for size in sizes for combo in combinations(methods, size)]
    if not variants:
        raise ValueError("Select methods or enable include_baseline")

    experiments = []
    for combo in variants:
        mitigation_files = [Path(mitigation_dir) / f"{m}.yaml"
                            for m in (combo or ("baseline",))]
        base = ExperimentConfig.from_files(**config_files, mitigations=mitigation_files)
        values = base.to_dict()
        for path, value in overrides.items():
            _set_value(values, path, value)

        grid = dict(hyperparameters)
        active_method_grid = {}
        for method in combo:
            for path, candidates in method_hyperparameters.get(method, {}).items():
                if path in active_method_grid and active_method_grid[path] != candidates:
                    raise ValueError(f"Conflicting method grids for {path} in {combo}")
                active_method_grid[path] = candidates
        grid.update(active_method_grid)
        variant = "_".join(combo) or "baseline"
        for index, candidates in enumerate(product(*grid.values()), start=1):
            for seed in seeds:
                resolved = deepcopy(values)
                for path, value in zip(grid, candidates):
                    _set_value(resolved, path, value)
                resolved["training"]["seed"] = seed
                resolved["evaluation"]["run_test"] = run_test
                experiments.append((
                    f"{name}_{variant}_hp{index:03d}_seed{seed}",
                    ExperimentConfig(resolved, list(base.sources)),
                ))
    return experiments


def experiment_plan_rows(experiments):
    """Flatten resolved settings so all values are visible before training."""
    def flatten(mapping, prefix=""):
        result = {}
        for key, value in mapping.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(flatten(value, path))
            else:
                result[path] = value
        return result

    return [{"experiment": name, **flatten(config.to_dict())}
            for name, config in experiments]
