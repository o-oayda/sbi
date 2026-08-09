"""Resolve inherited RACS experiment configurations."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import re
from typing import Any

import yaml


_EXPERIMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class ExperimentConfigError(ValueError):
    """Raised when an experiment configuration cannot be resolved."""


def _deep_merge(
    parent: Mapping[str, Any], child: Mapping[str, Any]
) -> dict[str, Any]:
    merged = deepcopy(dict(parent))
    for key, value in child.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_experiment_config(
    path: str | Path,
    *,
    experiment_dir: str | Path | None = None,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Resolve ``extends`` recursively and return the config plus source files."""
    selected = Path(path).resolve(strict=True)
    root = (
        Path(experiment_dir).resolve(strict=True)
        if experiment_dir is not None
        else selected.parent
    )

    def resolve(
        current: Path, stack: tuple[Path, ...]
    ) -> tuple[dict[str, Any], list[Path]]:
        current = current.resolve(strict=True)
        if current in stack:
            cycle = " -> ".join(item.stem for item in (*stack, current))
            raise ExperimentConfigError(f"Experiment inheritance cycle: {cycle}")
        try:
            with current.open(encoding="utf-8") as stream:
                raw = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            raise ExperimentConfigError(
                f"Invalid experiment config {current}: {error}"
            ) from error
        if not isinstance(raw, dict):
            raise ExperimentConfigError(
                f"Experiment config must contain a YAML mapping: {current}"
            )

        child = dict(raw)
        parent_name = child.pop("extends", None)
        if parent_name is None:
            return deepcopy(child), [current]
        if (
            not isinstance(parent_name, str)
            or _EXPERIMENT_NAME.fullmatch(parent_name) is None
        ):
            raise ExperimentConfigError(
                f"Invalid parent experiment name in {current}: {parent_name!r}"
            )
        parent_path = root / f"{parent_name}.yaml"
        if not parent_path.is_file():
            raise ExperimentConfigError(
                f"Parent experiment does not exist: {parent_name!r}"
            )
        parent, sources = resolve(parent_path, (*stack, current))
        return _deep_merge(parent, child), [*sources, current]

    resolved, sources = resolve(selected, ())
    return resolved, tuple(sources)
