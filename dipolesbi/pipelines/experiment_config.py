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


class ObservationConfigError(ValueError):
    """Raised when an observation configuration cannot be resolved."""


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


def _resolve_inherited_config(
    path: str | Path,
    *,
    config_dir: str | Path | None,
    config_kind: str,
    error_type: type[ValueError],
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Resolve one inheritance tree and return its merged mapping and sources."""
    selected = Path(path).resolve(strict=True)
    root = (
        Path(config_dir).resolve(strict=True)
        if config_dir is not None
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
            raise error_type(
                f"Invalid {config_kind} config {current}: {error}"
            ) from error
        if not isinstance(raw, dict):
            raise error_type(
                f"{config_kind.capitalize()} config must contain a YAML mapping: "
                f"{current}"
            )

        child = dict(raw)
        parent_name = child.pop("extends", None)
        if parent_name is None:
            return deepcopy(child), [current]
        if (
            not isinstance(parent_name, str)
            or _EXPERIMENT_NAME.fullmatch(parent_name) is None
        ):
            raise error_type(
                f"Invalid parent {config_kind} name in {current}: {parent_name!r}"
            )
        parent_path = root / f"{parent_name}.yaml"
        if not parent_path.is_file():
            raise error_type(
                f"Parent {config_kind} does not exist: {parent_name!r}"
            )
        parent, sources = resolve(parent_path, (*stack, current))
        return _deep_merge(parent, child), [*sources, current]

    resolved, sources = resolve(selected, ())
    return resolved, tuple(sources)


def resolve_experiment_config(
    path: str | Path,
    *,
    experiment_dir: str | Path | None = None,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Resolve an experiment ``extends`` tree."""
    return _resolve_inherited_config(
        path,
        config_dir=experiment_dir,
        config_kind="experiment",
        error_type=ExperimentConfigError,
    )


def resolve_observation_config(
    path: str | Path,
    *,
    observation_dir: str | Path | None = None,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Resolve an observation ``extends`` tree."""
    return _resolve_inherited_config(
        path,
        config_dir=observation_dir,
        config_kind="observation",
        error_type=ObservationConfigError,
    )
