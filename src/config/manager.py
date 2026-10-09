from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Any, Iterable

import yaml


class ConfigError(ValueError):
    """Raised when configuration files cannot be combined safely."""


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ConfigError(
            f"Top-level YAML structure must be a dictionary: {path}"
        )

    return data


def _deep_merge(
    base: dict[str, Any],
    incoming: dict[str, Any],
    *,
    source: str | Path | None = None,
    path: tuple[str, ...] = (),
) -> dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Rules:
    - Nested dictionaries are merged recursively.
    - Identical values are allowed.
    - Conflicting scalar/list values raise ConfigError.

    This prevents configuration values from being overwritten silently.
    """
    result = deepcopy(base)

    for key, incoming_value in incoming.items():
        current_path = path + (str(key),)

        if key not in result:
            result[key] = deepcopy(incoming_value)
            continue

        existing_value = result[key]

        if isinstance(existing_value, dict) and isinstance(incoming_value, dict):
            result[key] = _deep_merge(
                existing_value,
                incoming_value,
                source=source,
                path=current_path,
            )
            continue

        if existing_value == incoming_value:
            continue

        location = ".".join(current_path)
        source_message = f" in {source}" if source else ""

        raise ConfigError(
            f"Configuration conflict at '{location}'{source_message}: "
            f"{existing_value!r} != {incoming_value!r}"
        )

    return result


def _git_metadata() -> dict[str, Any]:
    """
    Return Git metadata for the current repository when available.

    If the project is not inside a Git repository, the values are set to None.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        return {
            "git_commit": commit,
            "git_dirty": bool(status),
        }

    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "git_commit": None,
            "git_dirty": None,
        }


def _normalize_paths(
    paths: Iterable[str | Path] | None,
) -> list[Path]:
    if paths is None:
        return []

    return [Path(path) for path in paths]


@dataclass
class ExperimentConfig:
    """
    Fully resolved configuration for one experimental condition.

    The usual setup is:
        - one model config
        - one training config
        - one evaluation config
        - one dataset config
        - zero or more mitigation configs
    """

    data: dict[str, Any]
    sources: list[Path]

    @classmethod
    def from_files(
        cls,
        *,
        model: str | Path,
        training: str | Path,
        dataset: str | Path,
        evaluation: str | Path | None = None,
        mitigations: Iterable[str | Path] | None = None,
    ) -> "ExperimentConfig":
        """
        Load and merge the configuration components of an experiment.

        Example
        -------
        config = ExperimentConfig.from_files(
            model="configs/model.yaml",
            training="configs/training.yaml",
            evaluation="configs/evaluation.yaml",
            dataset="configs/datasets/dataset_a.yaml",
            mitigations=[
                "configs/mitigation/balancing.yaml",
                "configs/mitigation/gender_augmentation.yaml",
            ],
        )
        """
        component_paths: list[Path] = [
            Path(model),
            Path(training),
        ]

        if evaluation is not None:
            component_paths.append(Path(evaluation))

        component_paths.append(Path(dataset))
        component_paths.extend(_normalize_paths(mitigations))

        merged: dict[str, Any] = {}

        for path in component_paths:
            config_part = _load_yaml(path)
            merged = _deep_merge(
                merged,
                config_part,
                source=path,
            )

        return cls(
            data=merged,
            sources=component_paths,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive copy of the resolved configuration."""
        return deepcopy(self.data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]


def _next_run_id(experiment_dir: Path) -> str:
    """
    Generate the next sequential run ID.

    Example:
        run_001
        run_002
        run_003
    """
    existing_numbers = []

    if experiment_dir.exists():
        for path in experiment_dir.iterdir():
            if not path.is_dir():
                continue

            if not path.name.startswith("run_"):
                continue

            try:
                existing_numbers.append(
                    int(path.name.removeprefix("run_"))
                )
            except ValueError:
                continue

    next_number = max(existing_numbers, default=0) + 1
    return f"run_{next_number:03d}"


def prepare_run(
    experiment_name: str,
    config: ExperimentConfig,
    *,
    results_root: str | Path = "results",
) -> tuple[ExperimentConfig, Path]:
    """
    Create a new run directory and save an immutable config snapshot.

    Structure
    ---------
    results/
        experiment_name/
            run_001/
                config.yaml

    The saved config contains:
        - the fully merged configuration
        - experiment name
        - run ID
        - timestamp
        - Git commit
        - Git dirty status
        - source config files

    Later modifications to configs/*.yaml therefore do not affect old runs.
    """
    results_root = Path(results_root)
    experiment_dir = results_root / experiment_name

    run_id = _next_run_id(experiment_dir)
    run_dir = experiment_dir / run_id

    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "experiment": experiment_name,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sources": [
            str(path) for path in config.sources
        ],
        **_git_metadata(),
    }

    resolved = config.to_dict()
    resolved["run"] = metadata

    snapshot_path = run_dir / "config.yaml"

    with snapshot_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            resolved,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    resolved_config = ExperimentConfig(
        data=resolved,
        sources=config.sources,
    )

    return resolved_config, run_dir


def load_run_config(
    run_directory: str | Path,
) -> ExperimentConfig:
    """
    Load the immutable configuration snapshot of an existing run.

    Example
    -------
    config = load_run_config(
        "results/balancing_dataset_a/run_001"
    )
    """
    run_directory = Path(run_directory)
    config_path = run_directory / "config.yaml"

    data = _load_yaml(config_path)

    return ExperimentConfig(
        data=data,
        sources=[config_path],
    )