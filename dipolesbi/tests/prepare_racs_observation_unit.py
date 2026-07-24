import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dipolesbi.pipelines import prepare_racs_observation as preparation
from dipolesbi.pipelines.racs_observation_helpers import load_reference_observation


def _observation_config(catalogue_path: Path, *, use_jax: bool = True):
    return {
        "catalogue_path": str(catalogue_path),
        "args": {
            "racs_epoch": "mid1",
            "flux_min": 15.0,
            "flux_temperature_min_mjy": 5.0,
            "fractional_error_flux_min_mjy": 10.0,
            "local_source_crossmatch_radius_arcsec": 5.0,
            "nside": 8,
            "downscale_nside": 4,
            "use_jax": use_jax,
            "chunk_size": 128,
            "temperature_model": "hot_linear",
            "paf_temperature_data_dir": "/tmp/paf",
            "openmeteo_fallback": False,
            "summary_features": ["log_dispersion", "flux_quantiles"],
            "flux_temperature_n_bins": 3,
            "flux_temperature_quantiles": [0.25, 0.5, 0.75],
            "flux_elevation_n_bins": 4,
            "flux_elevation_quantiles": [0.1, 0.9],
        },
        "mask": {
            "galactic_plane_width_deg": 5,
            "north_equatorial_pole_radius_deg": 42,
            "default_a_team_radius_deg": 2,
            "source_radii_deg": {"Cygnus A": 3, "LMC": 13, "SMC": 8},
        },
    }


def test_load_observation_config_requires_mapping(tmp_path):
    path = tmp_path / "observation.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        preparation.load_observation_config(path)


def test_load_observation_config_reads_yaml_mapping(tmp_path):
    path = tmp_path / "observation.yaml"
    expected = {"observation_id": "test", "args": {"nside": 8}}
    path.write_text(yaml.safe_dump(expected), encoding="utf-8")

    assert preparation.load_observation_config(path) == expected


def test_prepare_reference_observation_wires_config(tmp_path, monkeypatch):
    catalogue_path = tmp_path / "catalogue.fits"
    catalogue_path.touch()
    config = _observation_config(catalogue_path)
    calls = {}

    monkeypatch.setattr(
        preparation,
        "build_mask",
        lambda nside, **kwargs: calls.setdefault(
            "mask", (nside, kwargs)
        ) and np.ones(12 * nside**2, dtype=bool),
    )

    class FakeModelConfig:
        def __init__(self, **kwargs):
            calls["model_config"] = kwargs

    class FakeModel:
        def __init__(self, model_config):
            calls["model_type"] = "jax"
            calls["model"] = self

        def initialise_data(self):
            calls["initialised"] = True

    catalogue = object()
    monkeypatch.setattr(preparation, "RacsConfig", FakeModelConfig)
    monkeypatch.setattr(preparation, "RacsJax", FakeModel)
    monkeypatch.setattr(preparation, "load_catalogue", lambda path: catalogue)

    expected_x0 = np.arange(5, dtype=np.float32)
    expected_mask = np.ones(5, dtype=bool)

    def fake_build_real_sample(model, loaded_catalogue, flux_min, summaries, **kwargs):
        calls["real_sample"] = (
            model,
            loaded_catalogue,
            flux_min,
            summaries,
            kwargs,
        )
        return expected_x0, expected_mask

    monkeypatch.setattr(preparation, "build_real_sample", fake_build_real_sample)

    x0, mask = preparation.prepare_reference_observation(config)

    np.testing.assert_array_equal(x0, expected_x0)
    np.testing.assert_array_equal(mask, expected_mask)
    assert calls["initialised"] is True
    assert calls["model_config"]["catalogue_path"] == str(catalogue_path.resolve())
    assert calls["real_sample"][1] is catalogue
    assert calls["real_sample"][4]["local_source_crossmatch_radius_arcsec"] == 5.0
    assert calls["real_sample"][4]["save_map_plot"] is False


def test_save_reference_observation_round_trip(tmp_path):
    path = tmp_path / "nested" / "reference_observation.npz"
    x0 = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    mask = np.array([True, False, True])

    result = preparation.save_reference_observation(path, x0, mask)

    assert result == path
    with np.load(path, allow_pickle=False) as saved:
        assert saved.files == ["x0", "mask"]
        np.testing.assert_array_equal(saved["x0"], x0.astype(np.float32))
        np.testing.assert_array_equal(saved["mask"], mask)
        assert saved["mask"].dtype == np.bool_

    loaded_x0, loaded_mask = load_reference_observation(path)
    np.testing.assert_array_equal(loaded_x0, x0.astype(np.float32))
    np.testing.assert_array_equal(loaded_mask, mask)


def test_load_reference_observation_rejects_unexpected_keys(tmp_path):
    path = tmp_path / "reference.npz"
    np.savez(path, x0=np.zeros(2, dtype=np.float32), mask=np.ones(2, dtype=bool), extra=1)

    with pytest.raises(ValueError, match="must contain exactly"):
        load_reference_observation(path)


def test_load_reference_observation_requires_boolean_mask(tmp_path):
    path = tmp_path / "reference.npz"
    np.savez(
        path,
        x0=np.zeros(2, dtype=np.float32),
        mask=np.ones(2, dtype=np.int8),
    )

    with pytest.raises(ValueError, match="mask must have boolean dtype"):
        load_reference_observation(path)


def test_save_reference_observation_rejects_shape_mismatch(tmp_path):
    with pytest.raises(ValueError, match="identical shapes"):
        preparation.save_reference_observation(
            tmp_path / "reference.npz",
            np.zeros(3),
            np.ones(2, dtype=bool),
        )
