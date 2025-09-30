import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from functools import lru_cache
from joblib import Parallel, delayed
import numpy as np
import healpy as hp

from dipolesbi.catwise.maps import Catwise


@lru_cache(maxsize=1)
def _build_simulator() -> Catwise:
    sim = Catwise(cat_w1_max=17.0, cat_w12_min=0.5, use_float32=True)
    sim.initialise_data()
    return sim


def test_catwise_generate_dipole_single_run():
    sim = _build_simulator()

    n_samples = 1_000
    chunk_size = 128

    density_map, mask = sim.generate_dipole(
        n_initial_samples=n_samples,
        chunk_size=chunk_size,
        store_final_samples=False
    )

    n_pix = hp.nside2npix(sim.nside)
    assert density_map.shape == (n_pix,)
    assert mask.shape == (n_pix,)
    assert density_map.dtype == np.float32
    assert mask.dtype == np.bool_

    # Masked pixels should align with fill values (nan by default)
    assert np.all(np.isnan(density_map[~mask]))

    # Counts should be non-negative and not exceed the number of sampled sources
    unmasked_counts = density_map[mask]
    assert np.all(unmasked_counts >= 0)
    assert np.isfinite(unmasked_counts).all()
    assert np.nansum(density_map) <= n_samples


def test_catwise_generate_dipole_parallel_joblib():
    sim = _build_simulator()

    n_jobs = 2
    n_runs = 4
    n_samples = 500
    chunk_size = 128

    def _run_sim(seed: int):
        np.random.seed(seed)
        return sim.generate_dipole(
            n_initial_samples=n_samples,
            chunk_size=chunk_size,
            store_final_samples=False
        )

    outputs = Parallel(n_jobs=n_jobs)(
        delayed(_run_sim)(seed) for seed in range(n_runs)
    )

    density_stack = np.stack([out[0] for out in outputs], axis=0)
    mask_stack = np.stack([out[1] for out in outputs], axis=0)

    n_pix = hp.nside2npix(sim.nside)
    assert density_stack.shape == (n_runs, n_pix)
    assert mask_stack.shape == (n_runs, n_pix)

    # Binary mask should be identical across parallel simulations
    first_mask = mask_stack[0]
    assert np.all(mask_stack == first_mask)

    # Density entries should respect masking and remain finite for unmasked pixels
    assert np.all(np.isnan(density_stack[:, ~first_mask]))
    assert np.all(density_stack[:, first_mask] >= 0)


def test_add_error_statistical_properties():
    sim = _build_simulator()

    rng_seed = 1234
    n_samples = 200_000
    base_sigma = 0.05

    np.random.seed(rng_seed)
    w1 = np.full(n_samples, 13.5, dtype=np.float32)
    w2 = np.full(n_samples, 12.8, dtype=np.float32)

    w1_error = np.full(n_samples, base_sigma, dtype=np.float32)
    w2_error = np.full(n_samples, base_sigma, dtype=np.float32)

    # Gaussian baseline
    noisy_w1, noisy_w2 = sim.add_error(
        w1=(w1, w1_error),
        w2=(w2, w2_error),
        error_dist='gaussian'
    )
    residual_w1 = noisy_w1 - w1
    residual_w2 = noisy_w2 - w2

    assert abs(np.mean(residual_w1)) < 3e-3
    assert abs(np.mean(residual_w2)) < 3e-3

    var_w1 = np.var(residual_w1, ddof=1)
    var_w2 = np.var(residual_w2, ddof=1)
    expected_var = base_sigma ** 2
    assert np.isclose(var_w1, expected_var, rtol=0.1)
    assert np.isclose(var_w2, expected_var, rtol=0.1)

    # Gaussian with extra error (added in quadrature)
    w1_extra = 0.5
    w2_extra = 0.25
    np.random.seed(rng_seed)
    noisy_w1_extra, noisy_w2_extra = sim.add_error(
        w1=(w1, w1_error),
        w2=(w2, w2_error),
        w1_extra_error=w1_extra,
        w2_extra_error=w2_extra,
        error_dist='gaussian'
    )

    residual_w1_extra = noisy_w1_extra - w1
    residual_w2_extra = noisy_w2_extra - w2

    eff_sigma_w1 = np.sqrt(base_sigma ** 2 * (1 + w1_extra))
    eff_sigma_w2 = np.sqrt(base_sigma ** 2 * (1 + w2_extra))

    assert np.isclose(np.var(residual_w1_extra, ddof=1), eff_sigma_w1 ** 2, rtol=0.1)
    assert np.isclose(np.var(residual_w2_extra, ddof=1), eff_sigma_w2 ** 2, rtol=0.1)

    # Student's t distribution (nu from log10 params)
    log10_shape_param = np.log10(8.0)
    nu = 10 ** log10_shape_param

    np.random.seed(rng_seed)
    noisy_w1_t, noisy_w2_t = sim.add_error(
        w1=(w1, w1_error),
        w2=(w2, w2_error),
        error_dist='students-t',
        log10_shape_param=log10_shape_param
    )

    residual_w1_t = noisy_w1_t - w1
    residual_w2_t = noisy_w2_t - w2

    t_variance = expected_var * nu / (nu - 2)
    assert abs(np.mean(residual_w1_t)) < 5e-3
    assert abs(np.mean(residual_w2_t)) < 5e-3
    assert np.isclose(np.var(residual_w1_t, ddof=1), t_variance, rtol=0.2)
    assert np.isclose(np.var(residual_w2_t, ddof=1), t_variance, rtol=0.2)
