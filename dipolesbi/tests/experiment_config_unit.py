from pathlib import Path

import pytest
import yaml

from dipolesbi.pipelines.experiment_config import (
    ExperimentConfigError,
    resolve_experiment_config,
)


def _write(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def test_resolve_experiment_config_deep_merges_child_over_parent(tmp_path):
    _write(
        tmp_path / "fiducial.yaml",
        {
            "experiment_id": "fiducial",
            "execution": {"threads": 1, "mem_mb": 64_000},
            "args": {"n_rounds": 20, "simulate_clustering": "poisson"},
        },
    )
    _write(
        tmp_path / "noclus.yaml",
        {
            "extends": "fiducial",
            "experiment_id": "noclus",
            "args": {"simulate_clustering": None, "max_children": 0},
        },
    )

    resolved, sources = resolve_experiment_config(tmp_path / "noclus.yaml")

    assert resolved == {
        "experiment_id": "noclus",
        "execution": {"threads": 1, "mem_mb": 64_000},
        "args": {
            "n_rounds": 20,
            "simulate_clustering": None,
            "max_children": 0,
        },
    }
    assert [path.name for path in sources] == ["fiducial.yaml", "noclus.yaml"]


def test_resolve_experiment_config_rejects_missing_parent(tmp_path):
    _write(tmp_path / "child.yaml", {"extends": "missing"})

    with pytest.raises(ExperimentConfigError, match="does not exist"):
        resolve_experiment_config(tmp_path / "child.yaml")


def test_resolve_experiment_config_rejects_cycles(tmp_path):
    _write(tmp_path / "one.yaml", {"extends": "two"})
    _write(tmp_path / "two.yaml", {"extends": "one"})

    with pytest.raises(ExperimentConfigError, match="one -> two -> one"):
        resolve_experiment_config(tmp_path / "one.yaml")
