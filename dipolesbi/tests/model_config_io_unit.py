from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from catsim import RacsLow3Config

from dipolesbi.tools.model_config_io import load_model_config, save_model_config
from dipolesbi.tools.multiround_inferer import MultiRoundInferer


@dataclass
class ArrayConfig:
    values: np.ndarray


def _reference_observation_kwargs() -> dict[str, np.ndarray]:
    return {
        "reference_data": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "reference_mask": np.array([True, False, True], dtype=bool),
    }


def test_racs_low3_config_round_trip(tmp_path):
    config = RacsLow3Config(
        flux_min=15.0,
        nside=64,
        chunk_size=123,
        downscale_nside=32,
        paf_temperature_data_dir="/tmp/paf",
    )
    path = tmp_path / "model_config.json"

    save_model_config(config, path)
    loaded = load_model_config(path)

    assert loaded == config


def test_racs_low3_config_omits_mask_map_from_json(tmp_path):
    config = RacsLow3Config(
        flux_min=15.0,
        mask_map=np.ones(49152, dtype=bool),
    )
    path = tmp_path / "model_config.json"

    save_model_config(config, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "mask_map" not in payload["fields"]
    loaded = load_model_config(path)
    assert loaded.mask_map is None


def test_load_model_config_rejects_missing_class_name(tmp_path):
    path = tmp_path / "model_config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "class_module": "catsim.racs",
                "fields": {"flux_min": 15.0},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="class_name"):
        load_model_config(path)


def test_load_model_config_rejects_unknown_module(tmp_path):
    path = tmp_path / "model_config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "class_module": "not_a_real_module",
                "class_name": "Config",
                "fields": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModuleNotFoundError):
        load_model_config(path)


def test_load_model_config_rejects_unknown_class(tmp_path):
    path = tmp_path / "model_config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "class_module": "catsim.racs",
                "class_name": "NotAConfig",
                "fields": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AttributeError):
        load_model_config(path)


def test_load_model_config_rejects_non_dataclass_target(tmp_path):
    path = tmp_path / "model_config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "class_module": "builtins",
                "class_name": "dict",
                "fields": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="not a dataclass"):
        load_model_config(path)


def test_dump_configs_writes_model_config_json(tmp_path):
    config = RacsLow3Config(flux_min=15.0, nside=64, chunk_size=16)
    inferer = SimpleNamespace(
        mr_config=SimpleNamespace(plot_save_dir=str(tmp_path)),
        nflow_config="flow",
        nflow=None,
        train_config="train",
        transform_config="transform",
        model_config=config,
        initial_proposal="prior",
        **_reference_observation_kwargs(),
    )

    MultiRoundInferer._dump_configs(inferer)

    config_text = (tmp_path / "configs.txt").read_text(encoding="utf-8")
    assert "catsim.racs.RacsLow3Config" in config_text
    assert "'flux_min': 15.0" in config_text
    assert "'chunk_size': 16" in config_text
    assert load_model_config(tmp_path / "model_config.json") == config
    with np.load(tmp_path / "reference_observation.npz") as reference:
        np.testing.assert_array_equal(reference["x0"], inferer.reference_data)
        np.testing.assert_array_equal(reference["mask"], inferer.reference_mask)


def test_dump_configs_writes_ndarray_model_config(tmp_path):
    inferer = SimpleNamespace(
        mr_config=SimpleNamespace(plot_save_dir=str(tmp_path)),
        nflow_config="flow",
        nflow=None,
        train_config="train",
        transform_config="transform",
        model_config=ArrayConfig(np.arange(6, dtype=np.float32).reshape(2, 3)),
        initial_proposal="prior",
        **_reference_observation_kwargs(),
    )

    MultiRoundInferer._dump_configs(inferer)

    assert (tmp_path / "configs.txt").exists()
    loaded = load_model_config(tmp_path / "model_config.json")
    np.testing.assert_array_equal(loaded.values, inferer.model_config.values)
