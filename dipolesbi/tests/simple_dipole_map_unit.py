import numpy as np
import healpy as hp
import scipy.stats as sp_stats

from dipolesbi.tools.maps import SimpleDipoleMap
from dipolesbi.tools.np_rngkey import NPKey, poisson as poisson_sample
from dipolesbi.tools.healpix_helpers import downgrade_ignore_nan


def _make_theta(mean_density: float = 12.0) -> dict[str, np.ndarray]:
    return {
        "mean_density": np.array([mean_density], dtype=np.float64),
        "observer_speed": np.array([1.1], dtype=np.float64),
        "dipole_longitude": np.array([45.0], dtype=np.float64),
        "dipole_latitude": np.array([-30.0], dtype=np.float64),
    }


def test_generate_dipole_shapes_native_resolution() -> None:
    nside = 4
    npix = hp.nside2npix(nside)
    theta = _make_theta()

    mapper = SimpleDipoleMap(nside=nside, dtype=np.float64)
    rng_key = NPKey.from_seed(0)

    density_map, mask_map = mapper.generate_dipole(
        rng_key,
        theta,
        make_poisson_draws=False,
    )

    assert density_map.shape == (1, npix)
    assert mask_map.shape == (1, npix)
    np.testing.assert_array_equal(mask_map, np.ones_like(mask_map, dtype=np.bool_))

    expected_mean = mapper.dipole_signal(**theta)
    np.testing.assert_allclose(density_map, expected_mean)


def test_log_likelihood_matches_manual_native() -> None:
    nside = 4
    theta = _make_theta()

    base_mapper = SimpleDipoleMap(nside=nside, dtype=np.float64)
    model_signal = base_mapper.dipole_signal(**theta)

    rng_key = NPKey.from_seed(123)
    observed_counts = poisson_sample(
        rng_key,
        lam=model_signal,
        shape=model_signal.shape,
        dtype=np.int32,
    ).astype(np.float64)

    reference_mask = np.ones(model_signal.shape[1], dtype=np.bool_)
    reference_data = observed_counts[0]

    mapper = SimpleDipoleMap(
        nside=nside,
        dtype=np.float64,
        reference_data=reference_data,
        reference_mask=reference_mask,
    )

    logl = mapper.log_likelihood(theta)
    assert logl.shape == (model_signal.shape[0],)
    assert np.all(np.isfinite(logl))

    expected_logl = sp_stats.poisson.logpmf(
        k=reference_data[reference_mask],
        mu=model_signal[:, reference_mask],
    ).sum(axis=1)
    np.testing.assert_allclose(logl, expected_logl)


def test_log_likelihood_matches_manual_with_downscaling() -> None:
    nside = 4
    downscale_nside = 2
    theta = _make_theta(mean_density=20.0)

    base_mapper = SimpleDipoleMap(nside=nside, dtype=np.float64)
    model_signal = base_mapper.dipole_signal(**theta)

    fine_mask = np.ones(model_signal.shape[1], dtype=np.bool_)
    fine_mask[[0, 5, 23]] = False

    rng_key = NPKey.from_seed(7)
    observed_counts = poisson_sample(
        rng_key,
        lam=model_signal,
        shape=model_signal.shape,
        dtype=np.int32,
    ).astype(np.float64)

    coarse_counts, coarse_mask = downgrade_ignore_nan(
        observed_counts,
        fine_mask,
        downscale_nside,
    )
    coarse_counts = coarse_counts[0]
    coarse_mask = coarse_mask[0]

    mapper = SimpleDipoleMap(
        nside=nside,
        dtype=np.float64,
        reference_data=coarse_counts,
        reference_mask=fine_mask,
        downscale_nside=downscale_nside,
    )

    logl = mapper.log_likelihood(theta)
    assert logl.shape == (model_signal.shape[0],)
    assert np.all(np.isfinite(logl))

    model_mask = np.broadcast_to(fine_mask, model_signal.shape)
    coarse_signal, coarse_model_mask = downgrade_ignore_nan(
        model_signal,
        model_mask,
        downscale_nside,
    )
    coarse_model_mask = coarse_model_mask[0]

    assert np.array_equal(coarse_model_mask, coarse_mask)

    expected_logl = sp_stats.poisson.logpmf(
        k=coarse_counts[coarse_model_mask],
        mu=coarse_signal[:, coarse_model_mask],
    ).sum(axis=1)
    np.testing.assert_allclose(logl, expected_logl)


def test_log_likelihood_matches_manual_with_multi_step_downscaling() -> None:
    nside = 8
    downscale_nside = 2
    theta = _make_theta(mean_density=25.0)

    base_mapper = SimpleDipoleMap(nside=nside, dtype=np.float64)
    model_signal = base_mapper.dipole_signal(**theta)

    fine_mask = np.ones(model_signal.shape[1], dtype=np.bool_)
    fine_mask[[1, 12, 47, 120]] = False

    rng_key = NPKey.from_seed(21)
    observed_counts = poisson_sample(
        rng_key,
        lam=model_signal,
        shape=model_signal.shape,
        dtype=np.int32,
    ).astype(np.float64)

    coarse_counts, coarse_mask = downgrade_ignore_nan(
        observed_counts,
        fine_mask,
        downscale_nside,
    )
    coarse_counts = coarse_counts[0]
    coarse_mask = coarse_mask[0]

    mapper = SimpleDipoleMap(
        nside=nside,
        dtype=np.float64,
        reference_data=coarse_counts,
        reference_mask=fine_mask,
        downscale_nside=downscale_nside,
    )

    logl = mapper.log_likelihood(theta)
    assert logl.shape == (model_signal.shape[0],)
    assert np.all(np.isfinite(logl))

    model_mask = np.broadcast_to(fine_mask, model_signal.shape)
    coarse_signal, coarse_model_mask = downgrade_ignore_nan(
        model_signal,
        model_mask,
        downscale_nside,
    )
    coarse_model_mask = coarse_model_mask[0]

    assert np.array_equal(coarse_model_mask, coarse_mask)
    assert coarse_counts.shape[0] == 12 * downscale_nside * downscale_nside

    expected_logl = sp_stats.poisson.logpmf(
        k=coarse_counts[coarse_model_mask],
        mu=coarse_signal[:, coarse_model_mask],
    ).sum(axis=1)
    np.testing.assert_allclose(logl, expected_logl)
