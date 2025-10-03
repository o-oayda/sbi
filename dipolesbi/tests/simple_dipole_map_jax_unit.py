import numpy as np
import healpy as hp
import jax
import jax.numpy as jnp

from dipolesbi.tools.maps import SimpleDipoleMapJax
from dipolesbi.tools.healpix_helpers import downgrade_ignore_nan_jax


def _make_theta(mean_density: float = 12.0) -> dict[str, jnp.ndarray]:
    return {
        "mean_density": jnp.array(mean_density, dtype=jnp.float32),
        "observer_speed": jnp.array(1.1, dtype=jnp.float32),
        "dipole_longitude": jnp.array(45.0, dtype=jnp.float32),
        "dipole_latitude": jnp.array(-30.0, dtype=jnp.float32),
    }


def test_generate_dipole_shapes_and_means() -> None:
    nside = 4
    mapper = SimpleDipoleMapJax(nside=nside)
    theta = _make_theta()
    key = jax.random.PRNGKey(0)

    mean_counts = mapper.dipole_signal(**theta)
    npix = mapper.n_pix
    assert mean_counts.shape == (npix,)

    samples = mapper.generate_dipole(key, theta, make_poisson_draws=False)
    assert samples.shape == (npix,)
    np.testing.assert_allclose(np.array(samples), np.array(mean_counts))


def test_log_likelihood_native_resolution() -> None:
    nside = 4
    mapper = SimpleDipoleMapJax(nside=nside)
    theta = _make_theta(mean_density=20.0)

    key = jax.random.PRNGKey(42)
    mean_counts = mapper.dipole_signal(**theta)
    observed_counts = jax.random.poisson(key, mean_counts)

    mask = jnp.ones_like(mean_counts, dtype=bool)
    mapper = SimpleDipoleMapJax(
        nside=nside,
        reference_data=observed_counts,
        reference_mask=mask,
    )

    logl = mapper.log_likelihood(theta)
    assert logl.ndim == 0

    expected = jnp.sum(jax.scipy.stats.poisson.logpmf(k=observed_counts, mu=mean_counts))
    np.testing.assert_allclose(np.array(logl), np.array(expected))


def test_log_likelihood_with_downscaling() -> None:
    nside = 4
    downscale_nside = 2
    theta = _make_theta(mean_density=18.0)

    base_mapper = SimpleDipoleMapJax(nside=nside)
    mean_counts = base_mapper.dipole_signal(**theta)

    fine_mask = jnp.ones_like(mean_counts, dtype=bool)
    fine_mask = fine_mask.at[jnp.array([0, 5, 23])].set(False)

    key = jax.random.PRNGKey(17)
    observed_counts = jax.random.poisson(key, mean_counts)

    coarse_counts, coarse_mask = downgrade_ignore_nan_jax(
        observed_counts,
        fine_mask,
        downscale_nside
    )

    mapper = SimpleDipoleMapJax(
        nside=nside,
        reference_data=coarse_counts,
        reference_mask=fine_mask,
        downscale_nside=downscale_nside,
    )

    logl = mapper.log_likelihood(theta)
    assert logl.ndim == 0

    coarse_signal, coarse_model_mask = downgrade_ignore_nan_jax(
        mean_counts,
        fine_mask,
        downscale_nside
    )

    assert bool(jnp.all(coarse_model_mask == coarse_mask))

    expected = jnp.sum(
        jax.scipy.stats.poisson.logpmf(
            k=coarse_counts[coarse_mask],
            mu=coarse_signal[coarse_model_mask],
        )
    )
    np.testing.assert_allclose(np.array(logl), np.array(expected))


def test_log_likelihood_with_multi_step_downscaling() -> None:
    nside = 8
    downscale_nside = 2
    theta = _make_theta(mean_density=25.0)

    base_mapper = SimpleDipoleMapJax(nside=nside)
    mean_counts = base_mapper.dipole_signal(**theta)

    fine_mask = jnp.ones_like(mean_counts, dtype=bool)
    fine_mask = fine_mask.at[jnp.array([1, 12, 47, 120])].set(False)

    key = jax.random.PRNGKey(99)
    observed_counts = jax.random.poisson(key, mean_counts)

    coarse_counts, coarse_mask = downgrade_ignore_nan_jax(
        observed_counts,
        fine_mask,
        downscale_nside
    )

    mapper = SimpleDipoleMapJax(
        nside=nside,
        reference_data=coarse_counts,
        reference_mask=fine_mask,
        downscale_nside=downscale_nside,
    )

    logl = mapper.log_likelihood(theta)
    assert logl.ndim == 0

    coarse_signal, coarse_model_mask = downgrade_ignore_nan_jax(
        mean_counts,
        fine_mask,
        downscale_nside
    )

    assert coarse_counts.shape == (hp.nside2npix(downscale_nside),)
    assert coarse_signal.shape == coarse_counts.shape
    assert bool(jnp.all(coarse_model_mask == coarse_mask))

    expected = jnp.sum(
        jax.scipy.stats.poisson.logpmf(
            k=coarse_counts[coarse_mask],
            mu=coarse_signal[coarse_model_mask],
        )
    )
    np.testing.assert_allclose(np.array(logl), np.array(expected))
