import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import healpy as hp
import numpy as np
import pytest

from dipolesbi.scripts.based_racs_low3 import (
    build_racs_config,
    FLUX_TEMPERATURE_QUANTILES,
    _build_hybrid_sample_from_native,
    _flux_temperature_edges,
    _flux_temperature_quantile_features,
    _flux_temperature_quantile_ndim,
    _helmert_ilr_basis,
    _inverse_ilr_to_probabilities,
    _native_count_hist_features,
    _native_count_summary_features,
    make_simulator_wrapper,
)
from dipolesbi.tools.configs import (
    DataTransformConfig,
    MultiRoundInfererConfig,
    NeuralFlowConfig,
    ThetaTransformConfig,
    TransformConfig,
    TrainingConfig,
)
from dipolesbi.tools.multiround_inferer import MultiRoundInferer
from dipolesbi.tools.priors_np import DipolePriorNP


def _minimal_racs_config_kwargs(**overrides):
    kwargs = {
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
        "openmeteo_fallback": False,
    }
    kwargs.update(overrides)
    return kwargs


def test_build_racs_config_defaults_to_low3_product():
    config = build_racs_config(**_minimal_racs_config_kwargs())

    assert config.product.key == "low3"
    assert config.temperature_fallback == "none"
    assert config.store_final_samples is True


def test_build_racs_config_selects_mid1_product():
    config = build_racs_config(
        **_minimal_racs_config_kwargs(racs_epoch="mid1", use_jax=True)
    )

    assert config.product.key == "mid1"
    assert config.store_final_samples is False


def test_build_racs_config_enables_openmeteo_fallback():
    config = build_racs_config(
        **_minimal_racs_config_kwargs(openmeteo_fallback=True)
    )

    assert config.temperature_fallback == "open_meteo"


def test_native_count_hist_features_logs_normalized_counts():
    native_map = np.asarray([0, 1, 1, 2, 5, 9, 100, np.nan], dtype=np.float32)
    native_mask = np.asarray([True, True, True, False, True, True, True, True])

    features = _native_count_hist_features(
        native_map,
        native_mask,
        max_count=5,
        eps=1e-6,
    )

    expected_hist = np.asarray([1, 2, 0, 0, 0, 3], dtype=np.float64) / 6.0
    np.testing.assert_allclose(features, np.log(expected_hist + 1e-6), rtol=1e-6)
    assert features.dtype == np.float32


def test_helmert_ilr_basis_is_orthonormal_on_simplex():
    n_bins = 6
    basis = _helmert_ilr_basis(n_bins)

    assert basis.shape == (n_bins, n_bins - 1)
    np.testing.assert_allclose(basis.sum(axis=0), np.zeros(n_bins - 1), atol=1e-12)
    np.testing.assert_allclose(basis.T @ basis, np.eye(n_bins - 1), atol=1e-12)


def test_ilr_round_trip_recovers_positive_composition():
    probabilities = np.asarray([0.05, 0.15, 0.30, 0.20, 0.30], dtype=np.float64)
    basis = _helmert_ilr_basis(probabilities.size)

    z = np.log(probabilities) @ basis
    recovered = _inverse_ilr_to_probabilities(z, basis)

    np.testing.assert_allclose(recovered, probabilities, rtol=1e-12, atol=1e-12)


def test_native_count_log_dispersion_feature():
    native_map = np.asarray([0, 1, 1, 2, 5, np.nan], dtype=np.float32)
    native_mask = np.asarray([True, True, True, True, True, True])

    features = _native_count_summary_features(
        native_map,
        native_mask,
        max_count=5,
        eps=1e-6,
        summary="log_dispersion",
    )

    counts = np.asarray([0, 1, 1, 2, 5], dtype=np.float64)
    expected = np.log(np.var(counts, ddof=1) / np.mean(counts))
    np.testing.assert_allclose(features, np.asarray([expected], dtype=np.float32))
    assert features.dtype == np.float32


@pytest.mark.parametrize(
    ("transform", "expected_ndim"),
    [
        ("logprob", 6),
        ("ilr", 5),
    ],
)
def test_native_count_hist_feature_dimensions(transform, expected_ndim):
    native_map = np.asarray([0, 1, 1, 2, 5, 9, 100], dtype=np.float32)
    native_mask = np.ones_like(native_map, dtype=bool)

    features = _native_count_hist_features(
        native_map,
        native_mask,
        max_count=5,
        eps=1e-6,
        transform=transform,
    )

    assert features.shape == (expected_ndim,)
    assert np.isfinite(features).all()


def test_native_count_log_dispersion_feature_dimension():
    native_map = np.asarray([0, 1, 1, 2, 5, 9, 100], dtype=np.float32)
    native_mask = np.ones_like(native_map, dtype=bool)

    features = _native_count_summary_features(
        native_map,
        native_mask,
        max_count=5,
        eps=1e-6,
        summary="log_dispersion",
    )

    assert features.shape == (1,)
    assert np.isfinite(features).all()


def test_native_count_hist_features_rejects_empty_mask():
    native_map = np.asarray([0, 1, 2], dtype=np.float32)
    native_mask = np.zeros(3, dtype=bool)

    with pytest.raises(ValueError, match="no unmasked native pixels"):
        _native_count_hist_features(native_map, native_mask, max_count=5, eps=1e-6)


def test_flux_temperature_edges_use_finite_tile_temperature_range():
    class Model:
        tile_temperature_by_index = np.asarray([np.nan, 20.0, 25.0, 35.0])

    edges = _flux_temperature_edges(Model(), n_bins=3)

    np.testing.assert_allclose(edges, np.asarray([20.0, 25.0, 30.0, 35.0]))


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


def test_flux_temperature_quantile_features_rejects_empty_bins():
    with pytest.raises(ValueError, match="empty temperature bin"):
        _flux_temperature_quantile_features(
            observed_flux=np.asarray([1.0, 2.0]),
            temperature=np.asarray([1.0, 2.0]),
            temp_edges=np.asarray([0.0, 5.0, 10.0]),
        )


def test_flux_temperature_summary_dimension_defaults_to_fifty():
    assert _flux_temperature_quantile_ndim() == 10 * len(FLUX_TEMPERATURE_QUANTILES)


def test_simulator_wrapper_appends_flux_temperature_summary():
    class Model:
        downscale_nside = 1
        tile_temperature_by_index = np.linspace(0.0, 10.0, 11)
        final_observed_flux_samples = None
        final_temperature_samples = None

        def generate_dipole(self, *args, **kwargs):
            self.final_temperature_samples = np.arange(10, dtype=np.float32) + 0.5
            self.final_observed_flux_samples = np.arange(10, dtype=np.float32) + 1.0
            return (
                np.zeros(hp.nside2npix(1), dtype=np.float32),
                np.ones(hp.nside2npix(1), dtype=bool),
            )

    wrapper = make_simulator_wrapper(
        Model(),
        append_flux_temperature_summary=True,
    )

    data, mask = wrapper()

    map_ndim = hp.nside2npix(1)
    assert data.shape == (map_ndim + 50,)
    assert mask.shape == data.shape
    np.testing.assert_array_equal(mask[map_ndim:], np.ones(50, dtype=bool))
    expected_summary = np.repeat(np.arange(10, dtype=np.float32) + 1.0, 5)
    np.testing.assert_allclose(data[map_ndim:], expected_summary)


def test_simulator_wrapper_appends_native_and_flux_temperature_summaries():
    class Model:
        downscale_nside = 1
        tile_temperature_by_index = np.linspace(0.0, 10.0, 11)
        final_observed_flux_samples = None
        final_temperature_samples = None

        def generate_dipole(self, *args, **kwargs):
            self.final_temperature_samples = np.arange(10, dtype=np.float32) + 0.5
            self.final_observed_flux_samples = np.arange(10, dtype=np.float32) + 1.0
            native_map = np.arange(hp.nside2npix(2), dtype=np.float32)
            native_mask = np.ones(hp.nside2npix(2), dtype=bool)
            return native_map, native_mask

    wrapper = make_simulator_wrapper(
        Model(),
        append_native_count_summary=True,
        append_flux_temperature_summary=True,
        hist_max_count=4,
        native_count_summary="log_dispersion",
    )

    data, mask = wrapper()

    map_ndim = hp.nside2npix(1)
    assert data.shape == (map_ndim + 1 + 50,)
    assert mask.shape == data.shape
    assert bool(mask[map_ndim])
    np.testing.assert_array_equal(mask[map_ndim + 1 :], np.ones(50, dtype=bool))
    expected_flux_temperature = np.repeat(np.arange(10, dtype=np.float32) + 1.0, 5)
    np.testing.assert_allclose(data[map_ndim + 1 :], expected_flux_temperature)


def test_build_hybrid_sample_appends_all_true_summary_mask():
    native_nside = 2
    downscale_nside = 1
    native_map = np.arange(hp.nside2npix(native_nside), dtype=np.float32)
    native_mask = np.ones_like(native_map, dtype=bool)
    native_mask[:4] = False
    native_map[:4] = np.nan

    hybrid, hybrid_mask = _build_hybrid_sample_from_native(
        native_map,
        native_mask,
        downscale_nside=downscale_nside,
        hist_max_count=4,
        hist_eps=1e-6,
    )

    map_ndim = hp.nside2npix(downscale_nside)
    summary_ndim = 5
    assert hybrid.shape == (map_ndim + summary_ndim,)
    assert hybrid_mask.shape == hybrid.shape
    np.testing.assert_array_equal(
        hybrid_mask[map_ndim:],
        np.ones(summary_ndim, dtype=bool),
    )
    assert np.isfinite(hybrid[map_ndim:]).all()


def test_build_hybrid_sample_appends_all_true_ilr_summary_mask():
    native_nside = 2
    downscale_nside = 1
    native_map = np.arange(hp.nside2npix(native_nside), dtype=np.float32)
    native_mask = np.ones_like(native_map, dtype=bool)
    native_mask[:4] = False
    native_map[:4] = np.nan

    hybrid, hybrid_mask = _build_hybrid_sample_from_native(
        native_map,
        native_mask,
        downscale_nside=downscale_nside,
        hist_max_count=4,
        hist_eps=1e-6,
        native_count_hist_transform="ilr",
    )

    map_ndim = hp.nside2npix(downscale_nside)
    summary_ndim = 4
    assert hybrid.shape == (map_ndim + summary_ndim,)
    assert hybrid_mask.shape == hybrid.shape
    np.testing.assert_array_equal(
        hybrid_mask[map_ndim:],
        np.ones(summary_ndim, dtype=bool),
    )
    assert np.isfinite(hybrid[map_ndim:]).all()


def test_build_hybrid_sample_appends_log_dispersion_summary():
    native_nside = 2
    downscale_nside = 1
    native_map = np.arange(hp.nside2npix(native_nside), dtype=np.float32)
    native_mask = np.ones_like(native_map, dtype=bool)

    hybrid, hybrid_mask = _build_hybrid_sample_from_native(
        native_map,
        native_mask,
        downscale_nside=downscale_nside,
        hist_max_count=4,
        hist_eps=1e-6,
        native_count_summary="log_dispersion",
    )

    map_ndim = hp.nside2npix(downscale_nside)
    assert hybrid.shape == (map_ndim + 1,)
    assert hybrid_mask.shape == hybrid.shape
    assert bool(hybrid_mask[-1])
    assert np.isfinite(hybrid[-1])


def test_multiround_inferer_accepts_hybrid_target_dimension(tmp_path):
    map_ndim = hp.nside2npix(1)
    summary_ndim = 3
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
            plot_save_dir=str(tmp_path),
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


def test_multiround_config_accepts_native_count_summary():
    cfg = MultiRoundInfererConfig(
        simulation_budget=10,
        n_rounds=2,
        native_count_summary="log_dispersion",
    )

    assert cfg.native_count_summary == "log_dispersion"


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
