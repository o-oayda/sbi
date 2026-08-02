import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import healpy as hp
import numpy as np
import pytest
from astropy.table import Table
from catsim import RACS_PRODUCTS
from dipoleutils.utils.samples import CatalogueToMap

import dipolesbi.pipelines.racs_observation_helpers as observation_helpers
from dipolesbi.pipelines.based_racs import (
    DEFAULT_FLUX_ELEVATION_QUANTILES,
    DEFAULT_FLUX_TEMPERATURE_QUANTILES,
    build_hybrid_sample_from_native,
    build_prior_and_reference_theta,
    build_racs_config,
    build_scenario,
    load_inference_config,
    make_simulator_wrapper,
    _write_run_command,
)
from dipolesbi.pipelines.racs_observation_helpers import (
    _catalogue_view,
    build_mask,
    load_catalogue,
)
from dipolesbi.pipelines.summary_stats import (
    _flux_elevation_edges,
    _flux_elevation_quantile_features,
    _flux_elevation_quantile_ndim,
    _flux_temperature_edges,
    _flux_temperature_histogram_quantile_features,
    _flux_temperature_quantile_features,
    _flux_temperature_quantile_ndim,
    _native_count_log_dispersion_feature,
    _real_catalogue_flux_elevation_samples,
)
from dipolesbi.tools.configs import (
    DataTransformSpec,
    DataTransformConfig,
    MultiRoundInfererConfig,
    NeuralFlowConfig,
    Scenario,
    ThetaTransformConfig,
    TransformConfig,
    TrainingConfig,
)
from dipolesbi.tools.multiround_inferer import MultiRoundInferer
from dipolesbi.tools.priors_np import DipolePriorNP


def _minimal_racs_config_kwargs(**overrides):
    kwargs = {
        "catalogue_path": "/tmp/racs-test-catalogue.fits",
        "racs_epoch": "low3",
        "flux_min": 2.0,
        "nside": 1,
        "chunk_size": 16,
        "use_jax": False,
        "cluster_count_model": "geometric",
        "downscale_nside": None,
        "alpha_mean": 0.8,
        "alpha_sigma": 0.2,
        "fractional_error_flux_min_mjy": 10.0,
        "mask_map": np.ones(hp.nside2npix(1), dtype=bool),
        "max_cluster_children_per_parent": 16,
        "temperature_fallback": "none",
        "paf_temperature_data_dir": "/tmp/paf_temps",
    }
    kwargs.update(overrides)
    return kwargs


def test_build_mask_explicit_settings_match_defaults():
    default = build_mask(8)
    explicit = build_mask(
        8,
        galactic_plane_width_deg=5,
        north_equatorial_pole_radius_deg=42,
        default_a_team_radius_deg=2,
        source_radii_deg={"Cygnus A": 3, "LMC": 13, "SMC": 8},
    )

    np.testing.assert_array_equal(default, explicit)
    assert default.shape == (hp.nside2npix(8),)


def test_build_mask_config_allows_no_source_specific_radii(monkeypatch):
    calls = {}
    expected = np.ones(hp.nside2npix(8), dtype=bool)

    def fake_build_mask(nside, **kwargs):
        calls["nside"] = nside
        calls.update(kwargs)
        return expected

    monkeypatch.setattr(observation_helpers, "build_mask", fake_build_mask)
    config = {
        "args": {"nside": 8},
        "mask": {
            "galactic_plane_width_deg": 5,
            "north_equatorial_pole_radius_deg": 44,
            "default_a_team_radius_deg": 2,
        },
    }

    actual = observation_helpers.build_mask_from_observation_config(config)

    assert actual is expected
    assert calls["source_radii_deg"] == {}


def test_load_catalogue_reads_explicit_path(tmp_path):
    path = tmp_path / "catalogue.fits"
    expected = Table({"RA": [10.0, 20.0], "Dec": [-5.0, 6.0]})
    expected.write(path)

    loaded = load_catalogue(path)

    np.testing.assert_array_equal(loaded["RA"], expected["RA"])
    np.testing.assert_array_equal(loaded["Dec"], expected["Dec"])


def test_load_catalogue_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="Catalogue does not exist"):
        load_catalogue(tmp_path / "missing.fits")


def test_catalogue_view_forwards_local_crossmatch_radius(monkeypatch):
    calls = []
    catalogue = Table(
        {
            "Total_flux": [10.0],
            "Source_Name": ["RACS-source"],
        }
    )
    monkeypatch.setattr(CatalogueToMap, "make_cut", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        CatalogueToMap,
        "crossmatch_local_sources",
        lambda self, coordinate_system, radius, source_name_A_column: calls.append(
            (coordinate_system, radius, source_name_A_column)
        ),
    )

    _catalogue_view(
        catalogue,
        RACS_PRODUCTS["mid1"],
        minimum_flux=5.0,
        local_source_crossmatch_radius_arcsec=7.5,
    )

    assert calls == [("equatorial", 7.5, "Source_Name")]


def test_catalogue_view_can_disable_local_crossmatch(monkeypatch):
    catalogue = Table({"Total_flux": [10.0]})
    monkeypatch.setattr(CatalogueToMap, "make_cut", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        CatalogueToMap,
        "crossmatch_local_sources",
        lambda *args, **kwargs: pytest.fail("Cross-matching should be disabled."),
    )

    _catalogue_view(
        catalogue,
        RACS_PRODUCTS["mid1"],
        minimum_flux=5.0,
        local_source_crossmatch_radius_arcsec=None,
    )


def test_catalogue_view_requires_source_name_column_for_crossmatch(monkeypatch):
    catalogue = Table({"Total_flux": [10.0]})
    monkeypatch.setattr(CatalogueToMap, "make_cut", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="Source_Name.*missing"):
        _catalogue_view(
            catalogue,
            RACS_PRODUCTS["mid1"],
            minimum_flux=5.0,
            local_source_crossmatch_radius_arcsec=5.0,
        )


def test_build_racs_config_defaults_to_low3_product():
    config = build_racs_config(**_minimal_racs_config_kwargs())

    assert config.product.key == "low3"
    assert config.temperature_fallback == "none"
    assert config.temperature_model == "hot_linear"
    assert config.store_final_samples is True


def test_build_racs_config_selects_mid1_product():
    config = build_racs_config(
        **_minimal_racs_config_kwargs(racs_epoch="mid1", use_jax=True)
    )

    assert config.product.key == "mid1"
    assert config.store_final_samples is False


def test_build_racs_config_enables_openmeteo_fallback():
    config = build_racs_config(
        **_minimal_racs_config_kwargs(temperature_fallback="open_meteo")
    )

    assert config.temperature_fallback == "open_meteo"


def test_build_racs_config_wires_explicit_reference_fallback(monkeypatch):
    calls = {}

    class FakeRacsConfig:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr("dipolesbi.pipelines.based_racs.RacsConfig", FakeRacsConfig)

    build_racs_config(
        **_minimal_racs_config_kwargs(
            temperature_fallback="reference",
            paf_reference_temp_c=25.0,
            max_reference_fallback_tiles=1,
        )
    )

    assert calls["temperature_fallback"] == "reference"
    assert calls["paf_reference_temp_c"] == 25.0
    assert calls["max_reference_fallback_tiles"] == 1


def test_build_racs_config_accepts_paf_temperature_data_dir():
    config = build_racs_config(
        **_minimal_racs_config_kwargs(paf_temperature_data_dir="/tmp/paf_temps")
    )

    assert config.paf_temperature_data_dir == "/tmp/paf_temps"


def test_build_racs_config_accepts_independent_temperature_flux_cut():
    config = build_racs_config(
        **_minimal_racs_config_kwargs(
            flux_min=15.0,
            flux_temperature_min_mjy=2.0,
        )
    )

    assert config.flux_min == 15.0
    assert config.flux_temperature_min_mjy == 2.0


def test_build_racs_config_accepts_hot_quadratic_temperature_model():
    config = build_racs_config(
        **_minimal_racs_config_kwargs(temperature_model="hot_quadratic")
    )

    assert config.temperature_model == "hot_quadratic"


def test_write_run_command_records_shell_safe_python_invocation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dipolesbi.pipelines.based_racs.sys.orig_argv",
        ["python", "dipolesbi/pipelines/based_racs.py", "--out_dir", "run dir"],
    )

    command_path = _write_run_command(str(tmp_path / "run dir"))

    assert command_path.name == "run_command.txt"
    assert command_path.read_text(encoding="utf-8") == (
        "python dipolesbi/pipelines/based_racs.py --out_dir 'run dir'\n"
    )


def test_build_prior_and_reference_theta_accepts_custom_bounds():
    prior, theta_0 = build_prior_and_reference_theta(
        simulate_clustering="geometric",
        log10_n_initial_samples_range=(6.0, 7.0),
        observer_speed_range=(1.0, 9.0),
        dipole_longitude_range=(10.0, 250.0),
        dipole_latitude_range=(-30.0, 45.0),
        temp_beta_range=(0.01, 0.03),
        p_clus_range=(0.2, 0.8),
        clus_stop_prob_range=(0.5, 0.9),
    )

    assert prior.prior_dict["N"]["simulator_kwarg"] == "log10_n_initial_samples"
    assert prior.prior_dict["N"]["low_range"] == 6.0
    assert prior.prior_dict["N"]["high_range"] == 7.0
    assert prior.prior_dict["D"]["low_range"] == 1.0
    assert prior.prior_dict["D"]["high_range"] == 9.0
    assert prior.prior_dict["phi"]["low_range"] == 10.0
    assert prior.prior_dict["phi"]["high_range"] == 250.0
    assert prior.prior_dict["theta"]["low_range"] == -30.0
    assert prior.prior_dict["theta"]["high_range"] == 45.0
    assert prior.prior_dict["beta"]["low_range"] == 0.01
    assert prior.prior_dict["beta"]["high_range"] == 0.03
    assert prior.prior_dict["pclus"]["low_range"] == 0.2
    assert prior.prior_dict["pclus"]["high_range"] == 0.8
    assert prior.prior_dict["pstop"]["low_range"] == 0.5
    assert prior.prior_dict["pstop"]["high_range"] == 0.9
    assert theta_0["temp_beta"] == 0.02


def test_inference_yaml_reproduces_previous_nle_scenario():
    inference_config = load_inference_config(
        Path("workflow/configs/inference/nle_maf11_zscore.yaml")
    )
    prior, theta_0 = build_prior_and_reference_theta(
        simulate_clustering="poisson",
        log10_n_initial_samples_range=(5.6, 6.8),
        observer_speed_range=(0.0, 12.0),
        dipole_longitude_range=(0.0, 360.0),
        dipole_latitude_range=(-90.0, 90.0),
        temp_beta_range=(0.0, 0.05),
        lambda_clus_range=(0.0, 3.0),
    )
    common = {
        "nside": 4,
        "theta_prior": prior.to_jax(),
        "reference_theta": theta_0,
        "multiround_overrides": {
            "prng_integer_seed": 0,
            "plot_save_dir": "results/test",
            "simulation_budget": 100_000,
            "n_rounds": 20,
            "likelihood_chunk_size_gb": 0.5,
            "n_likelihood_samples": 10_000,
            "map_ndim": 192,
            "summary_ndim": 51,
            "native_count_summary": "log_dispersion",
        },
    }
    previous = Scenario.anynside_nle(
        **common,
        training_overrides={"learning_rate": 1e-4, "min_lr_ratio": 1.0},
        flow_overrides={
            "decoder_n_neurons": 128,
            "decoder_n_layers": 4,
            "architecture": 11 * ["MAF"],
            "data_reduction_factor": 0.5,
        },
        data_spec=DataTransformSpec.zscore(method="batchwise"),
    )
    configured = build_scenario(
        inference_config=inference_config,
        effective_nside=4,
        prior=prior,
        theta_0=theta_0,
        out_dir="results/test",
        ssnle_seed=0,
        n_rounds=20,
        n_simulations=100_000,
        map_ndim=192,
        summary_ndim=51,
        summary_features=["log_dispersion", "flux_quantiles"],
    )

    assert asdict(configured.training) == asdict(previous.training)
    assert asdict(configured.flow) == asdict(previous.flow)
    assert asdict(configured.multiround) == asdict(previous.multiround)
    assert asdict(configured.transforms.data_transform_config.spec) == asdict(
        previous.transforms.data_transform_config.spec
    )
    assert asdict(configured.transforms.theta_transform_config.spec) == asdict(
        previous.transforms.theta_transform_config.spec
    )


def test_native_count_log_dispersion_feature():
    native_map = np.asarray([0, 1, 1, 2, 5, np.nan], dtype=np.float32)
    native_mask = np.asarray([True, True, True, True, True, True])

    features = _native_count_log_dispersion_feature(native_map, native_mask)

    counts = np.asarray([0, 1, 1, 2, 5], dtype=np.float64)
    expected = np.log(np.var(counts, ddof=1) / np.mean(counts))
    np.testing.assert_allclose(features, np.asarray([expected], dtype=np.float32))
    assert features.dtype == np.float32


def test_native_count_log_dispersion_feature_dimension():
    native_map = np.asarray([0, 1, 1, 2, 5, 9, 100], dtype=np.float32)
    native_mask = np.ones_like(native_map, dtype=bool)

    features = _native_count_log_dispersion_feature(native_map, native_mask)

    assert features.shape == (1,)
    assert np.isfinite(features).all()


def test_native_count_log_dispersion_rejects_empty_mask():
    native_map = np.asarray([0, 1, 2], dtype=np.float32)
    native_mask = np.zeros(3, dtype=bool)

    with pytest.raises(ValueError, match="no unmasked native pixels"):
        _native_count_log_dispersion_feature(native_map, native_mask)


def test_flux_temperature_edges_use_finite_tile_temperature_range():
    class Model:
        tile_temperature_by_index = np.asarray([np.nan, 20.0, 25.0, 35.0])

    edges = _flux_temperature_edges(Model(), n_bins=3)

    np.testing.assert_allclose(edges, np.asarray([20.0, 25.0, 30.0, 35.0]))


def test_flux_temperature_edges_reject_invalid_bin_count():
    class Model:
        tile_temperature_by_index = np.asarray([20.0, 35.0])

    with pytest.raises(ValueError, match="at least one bin"):
        _flux_temperature_edges(Model(), n_bins=0)


def test_flux_temperature_quantile_features_by_temperature_bin():
    temp_edges = np.asarray([0.0, 10.0, 20.0], dtype=np.float64)
    flux = np.asarray([1.0, 3.0, 10.0, 30.0], dtype=np.float64)
    temperature = np.asarray([1.0, 9.0, 10.0, 20.0], dtype=np.float64)

    features = _flux_temperature_quantile_features(
        flux,
        temperature,
        temp_edges=temp_edges,
        quantiles=(0.0, 0.5, 1.0),
    )

    expected = np.asarray([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], dtype=np.float32)
    np.testing.assert_allclose(features, expected)
    assert features.dtype == np.float32


def test_flux_temperature_histogram_quantile_features_return_raw_flux_quantiles():
    temp_edges = np.asarray([0.0, 10.0, 20.0], dtype=np.float64)
    flux = np.asarray([10.0, 100.0, 1000.0, 10000.0], dtype=np.float64)
    temperature = np.asarray([1.0, 9.0, 10.0, 20.0], dtype=np.float64)

    features = _flux_temperature_histogram_quantile_features(
        flux,
        temperature,
        temp_edges=temp_edges,
        quantiles=(0.0, 0.5, 1.0),
        flux_min_mjy=10.0,
        flux_max_mjy=10000.0,
        n_flux_bins=3,
    )

    expected = np.asarray([10.0, 100.0, 1000.0, 1000.0, 3162.2777, 10000.0])
    np.testing.assert_allclose(features, expected, rtol=1e-5)
    assert features.dtype == np.float32


def test_flux_temperature_quantile_features_rejects_empty_bins():
    with pytest.raises(ValueError, match="empty temperature bin"):
        _flux_temperature_quantile_features(
            observed_flux=np.asarray([1.0, 2.0]),
            temperature=np.asarray([1.0, 2.0]),
            temp_edges=np.asarray([0.0, 5.0, 10.0]),
            quantiles=(0.5,),
        )


def test_flux_temperature_quantile_features_rejects_invalid_quantiles():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _flux_temperature_quantile_features(
            observed_flux=np.asarray([1.0, 2.0]),
            temperature=np.asarray([1.0, 2.0]),
            temp_edges=np.asarray([0.0, 3.0]),
            quantiles=(-0.1,),
        )


def test_flux_temperature_summary_dimension_defaults_to_fifty():
    assert _flux_temperature_quantile_ndim(
        10,
        DEFAULT_FLUX_TEMPERATURE_QUANTILES,
    ) == 50


def test_flux_temperature_summary_dimension_uses_custom_args():
    assert _flux_temperature_quantile_ndim(3, (0.25, 0.75)) == 6


def test_flux_elevation_edges_use_finite_lookup_range():
    class Model:
        elevation_lookup_values = np.asarray([np.nan, 20.0, 35.0, 50.0])

    edges = _flux_elevation_edges(Model(), n_bins=3)

    np.testing.assert_allclose(edges, np.asarray([20.0, 30.0, 40.0, 50.0]))


def test_flux_elevation_quantile_features_by_elevation_bin():
    features = _flux_elevation_quantile_features(
        observed_flux=np.asarray([1.0, 3.0, 10.0, 30.0]),
        elevation=np.asarray([1.0, 9.0, 10.0, 20.0]),
        elevation_edges=np.asarray([0.0, 10.0, 20.0]),
        quantiles=(0.0, 0.5, 1.0),
    )

    expected = np.asarray([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], dtype=np.float32)
    np.testing.assert_allclose(features, expected)


def test_flux_elevation_quantile_features_rejects_empty_bins():
    with pytest.raises(ValueError, match="empty elevation bin"):
        _flux_elevation_quantile_features(
            observed_flux=np.asarray([1.0, 2.0]),
            elevation=np.asarray([1.0, 2.0]),
            elevation_edges=np.asarray([0.0, 5.0, 10.0]),
            quantiles=(0.5,),
        )


def test_flux_elevation_summary_dimension_defaults_to_fifty():
    assert _flux_elevation_quantile_ndim(
        10,
        DEFAULT_FLUX_ELEVATION_QUANTILES,
    ) == 50


def test_real_catalogue_flux_elevation_samples_use_alt_and_sky_mask():
    catalogue = Table(
        {
            "RA": [0.0, 90.0, 180.0],
            "DEC": [0.0, 0.0, 0.0],
            "Total_flux": [10.0, 20.0, 30.0],
            "ALT": [40.0, 50.0, np.nan],
        }
    )
    pixels = hp.ang2pix(1, catalogue["RA"], catalogue["DEC"], lonlat=True, nest=True)
    mask = np.zeros(hp.nside2npix(1), dtype=bool)
    mask[pixels[0]] = True

    class Catalogue:
        def get_catalogue(self):
            return catalogue

    model = SimpleNamespace(
        nside=1,
        mask_map=mask,
        product=SimpleNamespace(
            label="RACS MID1",
            columns=SimpleNamespace(
                ra="RA",
                dec="DEC",
                total_flux="Total_flux",
                elevation="ALT",
            ),
        ),
    )

    flux, elevation = _real_catalogue_flux_elevation_samples(model, Catalogue())

    np.testing.assert_array_equal(flux, np.asarray([10.0]))
    np.testing.assert_array_equal(elevation, np.asarray([40.0]))


def test_simulator_wrapper_appends_flux_temperature_summary():
    class Model:
        downscale_nside = 1
        tile_temperature_by_index = np.linspace(0.0, 10.0, 11)
        def generate_dipole_with_flux_summaries(self, *args, **kwargs):
            return (
                np.zeros(hp.nside2npix(1), dtype=np.float32),
                np.ones(hp.nside2npix(1), dtype=bool),
                {
                    "temperature": np.repeat(
                        np.arange(10, dtype=np.float32) + 1.0,
                        5,
                    )
                },
            )

    wrapper = make_simulator_wrapper(
        Model(),
        summary_features=["flux_quantiles"],
    )

    data, mask = wrapper()

    map_ndim = hp.nside2npix(1)
    assert data.shape == (map_ndim + 50,)
    assert mask.shape == data.shape
    np.testing.assert_array_equal(mask[map_ndim:], np.ones(50, dtype=bool))
    expected_summary = np.repeat(np.arange(10, dtype=np.float32) + 1.0, 5)
    np.testing.assert_allclose(data[map_ndim:], expected_summary)


def test_simulator_wrapper_appends_log_dispersion_and_flux_quantiles():
    class Model:
        downscale_nside = 1
        tile_temperature_by_index = np.linspace(0.0, 10.0, 11)
        def generate_dipole_with_flux_summaries(self, *args, **kwargs):
            native_map = np.arange(hp.nside2npix(2), dtype=np.float32)
            native_mask = np.ones(hp.nside2npix(2), dtype=bool)
            return (
                native_map,
                native_mask,
                {
                    "temperature": np.repeat(
                        np.arange(10, dtype=np.float32) + 1.0,
                        5,
                    )
                },
            )

    wrapper = make_simulator_wrapper(
        Model(),
        summary_features=["log_dispersion", "flux_quantiles"],
    )

    data, mask = wrapper()

    map_ndim = hp.nside2npix(1)
    assert data.shape == (map_ndim + 1 + 50,)
    assert mask.shape == data.shape
    assert bool(mask[map_ndim])
    np.testing.assert_array_equal(mask[map_ndim + 1 :], np.ones(50, dtype=bool))
    expected_flux_temperature = np.repeat(np.arange(10, dtype=np.float32) + 1.0, 5)
    np.testing.assert_allclose(data[map_ndim + 1 :], expected_flux_temperature)


def test_simulator_wrapper_appends_flux_elevation_summary():
    class Model:
        downscale_nside = 1
        elevation_lookup_values = np.linspace(0.0, 10.0, 11)
        def generate_dipole_with_flux_summaries(self, *args, **kwargs):
            return (
                np.zeros(hp.nside2npix(1), dtype=np.float32),
                np.ones(hp.nside2npix(1), dtype=bool),
                {
                    "elevation": np.repeat(
                        np.arange(10, dtype=np.float32) + 1.0,
                        5,
                    )
                },
            )

    wrapper = make_simulator_wrapper(
        Model(),
        summary_features=["flux_elevation_quantiles"],
    )

    data, mask = wrapper()

    map_ndim = hp.nside2npix(1)
    assert data.shape == (map_ndim + 50,)
    np.testing.assert_array_equal(mask[map_ndim:], np.ones(50, dtype=bool))
    expected_summary = np.repeat(np.arange(10, dtype=np.float32) + 1.0, 5)
    np.testing.assert_allclose(data[map_ndim:], expected_summary)


def test_build_hybrid_sample_without_summaries_returns_map_only():
    native_nside = 2
    downscale_nside = 1
    native_map = np.arange(hp.nside2npix(native_nside), dtype=np.float32)
    native_mask = np.ones_like(native_map, dtype=bool)

    hybrid, hybrid_mask = build_hybrid_sample_from_native(
        native_map,
        native_mask,
        downscale_nside=downscale_nside,
        summary_features=[],
    )

    assert hybrid.shape == (hp.nside2npix(downscale_nside),)
    assert hybrid_mask.shape == hybrid.shape


def test_build_hybrid_sample_appends_log_dispersion_summary():
    native_nside = 2
    downscale_nside = 1
    native_map = np.arange(hp.nside2npix(native_nside), dtype=np.float32)
    native_mask = np.ones_like(native_map, dtype=bool)

    hybrid, hybrid_mask = build_hybrid_sample_from_native(
        native_map,
        native_mask,
        downscale_nside=downscale_nside,
        summary_features=["log_dispersion"],
    )

    map_ndim = hp.nside2npix(downscale_nside)
    assert hybrid.shape == (map_ndim + 1,)
    assert hybrid_mask.shape == hybrid.shape
    assert bool(hybrid_mask[-1])
    assert np.isfinite(hybrid[-1])


def test_multiround_inferer_accepts_hybrid_target_dimension(tmp_path):
    map_ndim = hp.nside2npix(1)
    summary_ndim = 3
    output_dir = tmp_path / "run"
    reference_data = np.zeros(map_ndim + summary_ndim, dtype=np.float32)
    reference_mask = np.ones_like(reference_data, dtype=bool)

    def simulator_function(*args, **kwargs):
        return reference_data[None, :], reference_mask[None, :]

    inferer = MultiRoundInferer(
        mode="NLE",
        initial_proposal=DipolePriorNP(),
        simulator_function=simulator_function,
        reference_observation=(reference_data, reference_mask),
        multi_round_config=MultiRoundInfererConfig(
            simulation_budget=2,
            n_rounds=1,
            plot_save_dir=str(output_dir),
            save_round_simulations=False,
            map_ndim=map_ndim,
            summary_ndim=summary_ndim,
        ),
        nflow_config=NeuralFlowConfig(mode="NLE", architecture=["MAF"]),
        transform_config=TransformConfig(
            data_transform_config=DataTransformConfig.zscore(),
            theta_transform_config=ThetaTransformConfig.blank_transform(),
        ),
        train_config=TrainingConfig(),
        use_ui=False,
    )

    assert inferer.data_ndim == map_ndim + summary_ndim
    assert inferer.map_ndim == map_ndim
    assert inferer.summary_start == map_ndim
    assert inferer.summary_ndim == summary_ndim
    assert inferer.nside == 1
    assert inferer.mr_config.plot_save_dir == str(output_dir)
    assert output_dir.is_dir()


def test_multiround_config_jax_ns_defaults_and_overrides():
    cfg = MultiRoundInfererConfig(simulation_budget=10, n_rounds=2)

    assert cfg.jax_ns_n_live == 5_000
    assert cfg.jax_ns_n_delete == 2_000

    cfg = MultiRoundInfererConfig(
        simulation_budget=10,
        n_rounds=2,
        jax_ns_n_live=2_000,
        jax_ns_n_delete=400,
    )

    assert cfg.jax_ns_n_live == 2_000
    assert cfg.jax_ns_n_delete == 400


@pytest.mark.parametrize(
    ("n_live", "n_delete", "match"),
    [
        (1, 1, "greater than 1"),
        (10, 0, "positive"),
        (10, 10, "less than jax_ns_n_live"),
        (10, 11, "less than jax_ns_n_live"),
    ],
)
def test_multiround_config_rejects_invalid_jax_ns_settings(
    n_live,
    n_delete,
    match,
):
    with pytest.raises(ValueError, match=match):
        MultiRoundInfererConfig(
            simulation_budget=10,
            n_rounds=2,
            jax_ns_n_live=n_live,
            jax_ns_n_delete=n_delete,
        )
