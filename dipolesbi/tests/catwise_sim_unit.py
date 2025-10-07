import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from functools import lru_cache
from joblib import Parallel, delayed
import numpy as np
import healpy as hp

from dipolesbi.catwise.maps import Catwise
from dipolesbi.tools.configs import CatwiseConfig
from dipolesbi.catwise.utils import AlphaLookup
from dipolesbi.tools.physics import (
    aberrate_points as fast_aberrate,
    rotation_matrices_for_dipole,
    native_to_dipole_frame,
    dipole_to_native_frame,
    compute_boosted_angles
)
from dipolesbi.tools.np_rngkey import prng_key
from dipolesbi.tools.utils import batch_simulate


@lru_cache(maxsize=1)
def _build_simulator() -> Catwise:
    cfg = CatwiseConfig(
        cat_w1_max=17.0,
        cat_w12_min=0.5,
        magnitude_error_dist='gaussian',
        use_float32=True,
    )
    sim = Catwise(cfg)
    sim.initialise_data()
    return sim


def test_catwise_generate_dipole_single_run():
    sim = _build_simulator()

    n_samples = 1_000

    density_map, mask = sim.generate_dipole(
        log10_n_initial_samples=np.log10(n_samples)
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

    def _run_sim(seed: int):
        np.random.seed(seed)
        return sim.generate_dipole(
            log10_n_initial_samples=np.log10(n_samples)
        )

    outputs = Parallel(n_jobs=n_jobs)(
        delayed(_run_sim)(seed) for seed in range(n_runs)
    )

    density_stack = np.stack([out[0] for out in outputs], axis=0) # pyright: ignore[reportOptionalSubscript]
    mask_stack = np.stack([out[1] for out in outputs], axis=0) # pyright: ignore[reportOptionalSubscript]
    

    n_pix = hp.nside2npix(sim.nside)
    assert density_stack.shape == (n_runs, n_pix)
    assert mask_stack.shape == (n_runs, n_pix)

    # Binary mask should be identical across parallel simulations
    first_mask = mask_stack[0]
    assert np.all(mask_stack == first_mask)

    # Density entries should respect masking and remain finite for unmasked pixels
    assert np.all(np.isnan(density_stack[:, ~first_mask]))
    assert np.all(density_stack[:, first_mask] >= 0)


def test_catwise_generate_dipole_reproducible_with_npkey():
    sim = _build_simulator()

    key = prng_key(1234)
    params = { # use a large-ish number to ensure maps aren't just full of zeros
        'log10_n_initial_samples': np.log10(1_000_000)
    }

    density_1, mask_1 = sim.generate_dipole(rng_key=key, **params)
    density_2, mask_2 = sim.generate_dipole(rng_key=key, **params)

    np.testing.assert_array_equal(mask_1, mask_2)
    np.testing.assert_array_equal(density_1, density_2)


def test_batch_simulate_reproducible_with_npkey():
    sim = _build_simulator()

    params = {
        'log10_n_initial_samples': np.full(3, np.log10(1_000_000), dtype=np.float32)
    }
    base_key = prng_key(99)

    sims_a, masks_a = batch_simulate(
        params,
        sim.generate_dipole,
        n_workers=2,
        rng_key=base_key
    )
    sims_b, masks_b = batch_simulate(
        params,
        sim.generate_dipole,
        n_workers=2,
        rng_key=base_key
    )

    np.testing.assert_array_equal(masks_a, masks_b)
    np.testing.assert_array_equal(sims_a, sims_b)


def test_batch_simulate_worker_invariance():
    sim = _build_simulator()

    params = {
        'log10_n_initial_samples': np.full(4, np.log10(1_000_000), dtype=np.float32)
    }
    base_key = prng_key(7)

    sims_serial, masks_serial = batch_simulate(
        params,
        sim.generate_dipole,
        n_workers=1,
        rng_key=base_key
    )
    sims_parallel, masks_parallel = batch_simulate(
        params,
        sim.generate_dipole,
        n_workers=3,
        rng_key=base_key
    )

    np.testing.assert_array_equal(masks_serial, masks_parallel)
    np.testing.assert_array_equal(sims_serial, sims_parallel)


def test_batch_simulate_key_diversity():
    sim = _build_simulator()

    params = {
        'log10_n_initial_samples': np.full(2, np.log10(1_000_000), dtype=np.float32)
    }
    key_a = prng_key(21)
    key_b = prng_key(22)

    sims_a, masks_a = batch_simulate(
        params,
        sim.generate_dipole,
        n_workers=2,
        rng_key=key_a
    )
    sims_b, masks_b = batch_simulate(
        params,
        sim.generate_dipole,
        n_workers=2,
        rng_key=key_b
    )

    np.testing.assert_array_equal(masks_a, masks_b)
    assert not np.array_equal(sims_a, sims_b)


def test_add_error_statistical_properties():
    sim = _build_simulator()

    rng_seed = 1234
    n_samples = 200_000
    base_sigma = 0.05

    rng = np.random.default_rng(rng_seed)
    sim.rng = np.random.default_rng(rng_seed)
    w1 = np.full(n_samples, 13.5, dtype=np.float32)
    w2 = np.full(n_samples, 12.8, dtype=np.float32)

    w1_error = np.full(n_samples, base_sigma, dtype=np.float32)
    w2_error = np.full(n_samples, base_sigma, dtype=np.float32)

    # Gaussian baseline
    noisy_w1, noisy_w2 = sim.add_error(
        w1=(w1, w1_error),
        w2=(w2, w2_error),
        error_dist='gaussian',
        rng=rng
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
    rng = np.random.default_rng(rng_seed)
    noisy_w1_extra, noisy_w2_extra = sim.add_error(
        w1=(w1, w1_error),
        w2=(w2, w2_error),
        w1_extra_error=w1_extra,
        w2_extra_error=w2_extra,
        error_dist='gaussian',
        rng=rng
    )

    residual_w1_extra = noisy_w1_extra - w1
    residual_w2_extra = noisy_w2_extra - w2

    eff_sigma_w1 = np.sqrt(base_sigma ** 2 * (1 + w1_extra))
    eff_sigma_w2 = np.sqrt(base_sigma ** 2 * (1 + w2_extra))

    assert np.isclose(np.var(residual_w1_extra, ddof=1), eff_sigma_w1 ** 2, rtol=0.1)
    assert np.isclose(np.var(residual_w2_extra, ddof=1), eff_sigma_w2 ** 2, rtol=0.1)

    # Common extra error flag should mirror W1 onto W2 when W2 not provided
    common_extra = 0.3
    rng = np.random.default_rng(rng_seed)
    noisy_w1_common, noisy_w2_common = sim.add_error(
        w1=(w1, w1_error),
        w2=(w2, w2_error),
        w1_extra_error=common_extra,
        w2_extra_error=None,
        common_extra_error=True,
        error_dist='gaussian',
        rng=rng
    )

    residual_w1_common = noisy_w1_common - w1
    residual_w2_common = noisy_w2_common - w2
    eff_sigma_common = np.sqrt(base_sigma ** 2 * (1 + common_extra))

    assert np.isclose(np.var(residual_w1_common, ddof=1), eff_sigma_common ** 2, rtol=0.1)
    assert np.isclose(np.var(residual_w2_common, ddof=1), eff_sigma_common ** 2, rtol=0.1)

    # Student's t distribution (nu from log10 params)
    log10_shape_param = np.log10(8.0)
    nu = 10 ** log10_shape_param

    rng = np.random.default_rng(rng_seed)
    noisy_w1_t, noisy_w2_t = sim.add_error(
        w1=(w1, w1_error),
        w2=(w2, w2_error),
        error_dist='students-t',
        log10_shape_param=log10_shape_param,
        rng=rng
    )

    residual_w1_t = noisy_w1_t - w1
    residual_w2_t = noisy_w2_t - w2

    t_variance = expected_var * nu / (nu - 2)
    assert abs(np.mean(residual_w1_t)) < 5e-3
    assert abs(np.mean(residual_w2_t)) < 5e-3
    assert np.isclose(np.var(residual_w1_t, ddof=1), t_variance, rtol=0.2)
    assert np.isclose(np.var(residual_w2_t, ddof=1), t_variance, rtol=0.2)


def _build_downscaled_simulator(nside_out: int) -> Catwise:
    cfg = CatwiseConfig(
        cat_w1_max=17.0,
        cat_w12_min=0.5,
        magnitude_error_dist='gaussian',
        use_float32=True,
        downscale_nside=nside_out,
    )
    sim = Catwise(cfg)
    sim.initialise_data()
    return sim


def test_generate_dipole_downscaled_shapes():
    coarse_nside = 32
    sim = _build_downscaled_simulator(coarse_nside)

    n_samples = 2_000
    dmap, mask = sim.generate_dipole(
        log10_n_initial_samples=np.log10(n_samples)
    )

    coarse_npix = hp.nside2npix(coarse_nside)
    native_npix = hp.nside2npix(sim.nside)

    assert dmap.shape == (coarse_npix,)
    assert mask.shape == (coarse_npix,)
    assert dmap.dtype == np.float32
    assert mask.dtype == np.bool_

    assert np.all(np.isnan(dmap[~mask]))
    assert np.all(dmap[mask] >= 0)
    assert np.nansum(dmap) <= n_samples

    # properties should reuse the cached downscaled outputs
    assert np.array_equal(sim.density_map, dmap, equal_nan=True)
    assert np.array_equal(sim.binary_mask, mask)

    native_mask = sim.native_mask
    assert native_mask.shape == (native_npix,)
    assert native_mask.dtype == np.bool_


def test_make_real_sample_downscaled_shapes():
    coarse_nside = 32
    sim = _build_downscaled_simulator(coarse_nside)

    real_map, real_mask = sim.make_real_sample()

    coarse_npix = hp.nside2npix(coarse_nside)

    assert real_map.shape == (coarse_npix,)
    assert real_mask.shape == (coarse_npix,)
    assert real_map.dtype == np.float32
    assert real_mask.dtype == np.bool_

    assert np.all(np.isnan(real_map[~real_mask]))
    assert np.array_equal(sim.real_density_map, real_map, equal_nan=True)

def test_alpha_lookup_buffered_vs_default():
    lookup = AlphaLookup(no_check=True)

    colours = np.linspace(-1.0, 2.0, 1_000, dtype=np.float32)
    out_buffer = np.empty_like(colours)

    buffered = lookup.fit_alpha(colours, out=out_buffer)

    # Recompute using a fresh lookup to avoid buffer reuse
    lookup_fresh = AlphaLookup(no_check=True)
    reference = lookup_fresh.p_W12(colours).astype(np.float32)

    assert buffered is out_buffer
    assert np.allclose(buffered, reference, rtol=1e-6, atol=1e-6)


def test_aberrate_points_matches_astropy_reference():
    rng = np.random.default_rng(123)
    longitudes = rng.uniform(0, 360, size=10_000)
    latitudes = np.degrees(np.arcsin(rng.uniform(-1, 1, size=10_000))) # (-90, 90)

    observer_direction = (123.4, -27.5)
    observer_speed = 1.23e-3

    rotation_matrices = rotation_matrices_for_dipole(*observer_direction)
    fast_lon, fast_lat, fast_theta = fast_aberrate(
        rest_longitudes=longitudes,
        rest_latitudes=latitudes,
        observer_direction=observer_direction,
        observer_speed=observer_speed,
        rotation_matrices=rotation_matrices
    )

    ref_lon_dp, ref_lat_dp = native_to_dipole_frame(
        point_longitudes=longitudes,
        point_latitudes=latitudes,
        dipole_longitude=observer_direction[0],
        dipole_latitude=observer_direction[1]
    )
    ref_theta = 90.0 - ref_lat_dp
    boosted_theta = compute_boosted_angles(
        source_frame_angles=ref_theta,
        observer_speed=observer_speed
    )
    boosted_lat_dp = 90.0 - boosted_theta
    ref_lon, ref_lat = dipole_to_native_frame(
        point_longitudes=ref_lon_dp,
        point_latitudes=boosted_lat_dp,
        dipole_longitude=observer_direction[0],
        dipole_latitude=observer_direction[1]
    )

    assert np.allclose(fast_lon, ref_lon, atol=1e-9)
    assert np.allclose(fast_lat, ref_lat, atol=1e-9)
    assert np.allclose(fast_theta, ref_theta, atol=1e-9)
