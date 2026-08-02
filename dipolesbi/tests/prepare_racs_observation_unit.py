import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dipolesbi.pipelines import prepare_racs_observation as preparation
from dipolesbi.pipelines.racs_observation_helpers import load_reference_observation


def _observation_config(*, use_jax: bool = True):
    return {
        "datasets": {"catalogue": "test_catalogue"},
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
            "temperature_fallback": "none",
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
    paf_temperature_data_dir = tmp_path / "paf_temps"
    paf_temperature_data_dir.mkdir()
    config = _observation_config()
    calls = {}

    monkeypatch.setattr(
        preparation,
        "build_mask_from_observation_config",
        lambda observation_config: calls.setdefault(
            "mask_config", observation_config
        ) and np.ones(12 * observation_config["args"]["nside"] ** 2, dtype=bool),
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

    x0, mask = preparation.prepare_reference_observation(
        config,
        catalogue_path,
        paf_temperature_data_dir,
    )

    np.testing.assert_array_equal(x0, expected_x0)
    np.testing.assert_array_equal(mask, expected_mask)
    assert calls["initialised"] is True
    assert calls["mask_config"] is config
    assert calls["model_config"]["catalogue_path"] == str(catalogue_path.resolve())
    assert calls["model_config"]["paf_temperature_data_dir"] == str(
        paf_temperature_data_dir.resolve()
    )
    assert "temperature_model" not in calls["model_config"]
    assert calls["real_sample"][1] is catalogue
    assert calls["real_sample"][4]["local_source_crossmatch_radius_arcsec"] == 5.0
    assert calls["real_sample"][4]["save_map_plot"] is False


def test_prepare_reference_observation_accepts_no_paf_directory_for_open_meteo(
    tmp_path, monkeypatch
):
    catalogue_path = tmp_path / "catalogue.fits"
    catalogue_path.touch()
    config = _observation_config()
    config["args"]["racs_epoch"] = "low2"
    config["args"]["temperature_fallback"] = "open_meteo"
    calls = {}

    monkeypatch.setattr(
        preparation,
        "build_mask_from_observation_config",
        lambda config: np.ones(12 * config["args"]["nside"] ** 2, dtype=bool),
    )

    class FakeModelConfig:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    class FakeModel:
        def __init__(self, config):
            pass

        def initialise_data(self):
            pass

    monkeypatch.setattr(preparation, "RacsConfig", FakeModelConfig)
    monkeypatch.setattr(preparation, "RacsJax", FakeModel)
    monkeypatch.setattr(preparation, "load_catalogue", lambda path: object())
    monkeypatch.setattr(
        preparation,
        "build_real_sample",
        lambda *args, **kwargs: (np.zeros(1), np.ones(1, dtype=bool)),
    )

    preparation.prepare_reference_observation(config, catalogue_path, None)

    assert calls["paf_temperature_data_dir"] is None
    assert calls["temperature_fallback"] == "open_meteo"


def test_prepare_reference_observation_rejects_no_paf_without_low2_fallback(
    tmp_path,
):
    catalogue_path = tmp_path / "catalogue.fits"
    catalogue_path.touch()

    with pytest.raises(ValueError, match="may only be omitted for LOW2"):
        preparation.prepare_reference_observation(
            _observation_config(), catalogue_path, None
        )


def test_prepare_reference_observation_wires_explicit_reference_fallback(
    tmp_path, monkeypatch
):
    catalogue_path = tmp_path / "catalogue.fits"
    catalogue_path.touch()
    paf_temperature_data_dir = tmp_path / "paf_temps"
    paf_temperature_data_dir.mkdir()
    config = _observation_config()
    config["args"].update(
        temperature_fallback="reference",
        paf_reference_temp_c=25.0,
        max_reference_fallback_tiles=1,
    )
    calls = {}

    monkeypatch.setattr(
        preparation,
        "build_mask_from_observation_config",
        lambda config: np.ones(12 * config["args"]["nside"] ** 2, dtype=bool),
    )

    class FakeModelConfig:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    class FakeModel:
        def __init__(self, config):
            pass

        def initialise_data(self):
            pass

    monkeypatch.setattr(preparation, "RacsConfig", FakeModelConfig)
    monkeypatch.setattr(preparation, "RacsJax", FakeModel)
    monkeypatch.setattr(preparation, "load_catalogue", lambda path: object())
    monkeypatch.setattr(
        preparation,
        "build_real_sample",
        lambda *args, **kwargs: (np.zeros(1), np.ones(1, dtype=bool)),
    )

    preparation.prepare_reference_observation(
        config, catalogue_path, paf_temperature_data_dir
    )

    assert calls["temperature_fallback"] == "reference"
    assert calls["paf_reference_temp_c"] == 25.0
    assert calls["max_reference_fallback_tiles"] == 1


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            {"paf_reference_temp_c": np.nan, "max_reference_fallback_tiles": 1},
            "finite paf_reference_temp_c",
        ),
        (
            {"paf_reference_temp_c": 25.0, "max_reference_fallback_tiles": 0},
            "positive max_reference_fallback_tiles",
        ),
    ],
)
def test_prepare_reference_observation_rejects_unsafe_reference_fallback(
    tmp_path, settings, message
):
    catalogue_path = tmp_path / "catalogue.fits"
    catalogue_path.touch()
    paf_temperature_data_dir = tmp_path / "paf_temps"
    paf_temperature_data_dir.mkdir()
    config = _observation_config()
    config["args"].update(temperature_fallback="reference", **settings)

    with pytest.raises(ValueError, match=message):
        preparation.prepare_reference_observation(
            config, catalogue_path, paf_temperature_data_dir
        )


def test_prepare_reference_observation_can_include_native_map(tmp_path, monkeypatch):
    catalogue_path = tmp_path / "catalogue.fits"
    catalogue_path.touch()
    paf_temperature_data_dir = tmp_path / "paf_temps"
    paf_temperature_data_dir.mkdir()
    calls = []

    monkeypatch.setattr(
        preparation,
        "build_mask_from_observation_config",
        lambda config: np.ones(12 * config["args"]["nside"] ** 2, dtype=bool),
    )
    monkeypatch.setattr(preparation, "RacsConfig", lambda **kwargs: object())

    class FakeModel:
        def __init__(self, config):
            self.downscale_nside = 4

        def initialise_data(self):
            pass

    monkeypatch.setattr(preparation, "RacsJax", FakeModel)
    monkeypatch.setattr(preparation, "load_catalogue", lambda path: "catalogue")

    def fake_build_real_sample(model, catalogue, flux_min, summaries, **kwargs):
        calls.append((model.downscale_nside, summaries, kwargs))
        size = 4 if model.downscale_nside == 4 else 12
        return np.arange(size, dtype=np.float32), np.ones(size, dtype=bool)

    monkeypatch.setattr(preparation, "build_real_sample", fake_build_real_sample)

    prepared = preparation.prepare_reference_observation(
        _observation_config(),
        catalogue_path,
        paf_temperature_data_dir,
        include_native=True,
    )

    x0, mask, native_x0, native_mask = prepared
    assert x0.shape == mask.shape == (4,)
    assert native_x0.shape == native_mask.shape == (12,)
    assert calls[0][0] == 4
    assert calls[0][1] == ["log_dispersion", "flux_quantiles"]
    assert calls[1] == (
        None,
        [],
        {
            "local_source_crossmatch_radius_arcsec": 5.0,
            "save_map_plot": False,
        },
    )


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


def test_save_reference_observation_preserves_unchanged_file_mtime(tmp_path):
    path = tmp_path / "reference_observation.npz"
    x0 = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    mask = np.array([True, False, True])
    preparation.save_reference_observation(path, x0, mask)
    old_mtime_ns = 1_000_000_000
    os.utime(path, ns=(old_mtime_ns, old_mtime_ns))

    preparation.save_reference_observation(path, x0, mask)

    assert path.stat().st_mtime_ns == old_mtime_ns


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
