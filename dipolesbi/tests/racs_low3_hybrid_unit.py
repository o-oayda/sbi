import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import healpy as hp
import numpy as np
import pytest

from dipolesbi.scripts.based_racs_low3 import (
    _build_hybrid_sample_from_native,
    _native_count_hist_features,
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


def test_native_count_hist_features_rejects_empty_mask():
    native_map = np.asarray([0, 1, 2], dtype=np.float32)
    native_mask = np.zeros(3, dtype=bool)

    with pytest.raises(ValueError, match="no unmasked native pixels"):
        _native_count_hist_features(native_map, native_mask, max_count=5, eps=1e-6)


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
