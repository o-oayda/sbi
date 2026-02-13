from functools import partial
import os
from typing import Any, Literal
from blackjax.types import Array
from catsim import Catwise, CatwiseConfig
from catsim.simulator import downgrade_ignore_nan
from numpy.typing import NDArray
from dipolesbi.tools.jax_ns import run_ns_from_chkpt
import jax
import numpy as np
import argparse
import healpy as hp
import matplotlib.pyplot as plt
from dipolesbi.tools.priors_jax import DipolePriorJax
import jax.numpy as jnp
import jax.scipy as jsp
from dipolesbi.tools.coordinates import (
    _galactic_to_equatorial_vec_jax, _ang2vec_jax
)


EXPECTED_AMPLITUDES = {
    'nvss': 4.31e-3,
    'racs': 4.27e-3,
    'catwise': 7.25e-3,
    'planck': 0.0012336
}

def lnlike_radio(
        params: dict[str, Any],
        expected_amplitude: float,
        pixel_counts: NDArray,
        pixel_vectors: NDArray
) -> Array:
    D = params['observer_speed'] * expected_amplitude
    Nbar = params['mean_density']
    gal_lon, gal_lat = params['dipole_longitude'], params['dipole_latitude']
    dipole_vector_eq = _galactic_to_equatorial_vec_jax(
        jnp.asarray(gal_lon),
        jnp.asarray(gal_lat)
    )

    pixel_vectors = jnp.asarray(pixel_vectors) # pyright: ignore[reportAssignmentType]
    pixel_counts = jnp.asarray(pixel_counts)   # pyright: ignore[reportAssignmentType]
    # (3,), (3, 49152)
    expected_number_density_list = Nbar * (
        1. + D * jnp.einsum('i,ij', dipole_vector_eq, pixel_vectors)
    )

    return jnp.sum(
        jsp.stats.poisson.logpmf(pixel_counts, expected_number_density_list)
    )

def lnlike_planck(
        params: dict[str, Any],
        expected_amplitude: float,
        pixel_counts: NDArray,
        pixel_vectors: NDArray
) -> Array:
    D = params['observer_speed'] * expected_amplitude
    Nbar = params['mean_density']
    gal_lon, gal_lat = params['dipole_longitude'], params['dipole_latitude']
    # planck map is in Galactuc coordinates
    dipole_vector_eq = _ang2vec_jax(gal_lon, gal_lat)

    pixel_vectors = jnp.asarray(pixel_vectors) # pyright: ignore[reportAssignmentType]
    pixel_counts = jnp.asarray(pixel_counts)   # pyright: ignore[reportAssignmentType]
    # (3,), (3, 49152)
    expected_number_density_list = Nbar * (
        1. + D * jnp.einsum('i,ij', dipole_vector_eq, pixel_vectors)
    )

    return jnp.sum(
        jsp.stats.norm.logpdf(
            pixel_counts, expected_number_density_list, scale=planck_std
        )
    )

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--joint-sample',
        choices=['nvss', 'racs', 'planck']
    )
    args = argparser.parse_args()

    DOWNSCALE_NSIDE = 4
    CATWISE_VERSION: Literal['S21', 'S22'] = 'S22'
    PATH_TO_CHKPT = 'S22_NLE/20260212_205211_SEED0_NLE/nflow_checkpoint.npz' # wide v prior
    PRNG_SEED = 42
    JOINT_SAMPLE = args.joint_sample
    JOINT_NSIDE = 64

    config = CatwiseConfig(
        cat_w1_max=17.0, 
        cat_w12_min=0.5,
        magnitude_error_dist='gaussian',
        downscale_nside=DOWNSCALE_NSIDE,
        base_mask_version=CATWISE_VERSION,
        s21_catalogue_path=(
            '~/Documents/catsim/src/catsim/data/'
            'catwise_agns_masked_final_w1lt16p5_alpha.fits'
        )
    )

    model = Catwise(config)
    model.initialise_data()
    pixel_vectors = hp.pix2vec(
        nside=JOINT_NSIDE,
        ipix=np.arange(hp.nside2npix(JOINT_NSIDE))
    )
    xyz = np.vstack(pixel_vectors)
    expected_amplitude = EXPECTED_AMPLITUDES[JOINT_SAMPLE]

    if JOINT_SAMPLE == 'nvss':
        sample_B = np.load('maps/nvssb_dmap.npy')
        # hp.projview(sample_B)
        # plt.show()
    elif JOINT_SAMPLE == 'racs':
        # so the joint log z does seem to work --- racs dipole poisson lnZ ~ 91195,
        # catsim lnZ ~ 757, the sum of these ~ 91955, which is what we get from
        # the joint log Z. So the scale looks right, but it should not generally
        # be the case that Z_AB = Z_A Z_B
        sample_B = np.load('maps/racsb_dmap.npy')
        # hp.projview(sample_B)
        # plt.show()
    elif JOINT_SAMPLE == 'planck':
        sample_B = np.load('maps/planck_galactic_counts.npy')
        planck_std = np.std(sample_B)
    else:
        raise Exception

    mask = ~np.isnan(sample_B)
    lnlike_B = partial(
        lnlike_radio if JOINT_SAMPLE in ['racs', 'nvss'] else lnlike_planck,
        expected_amplitude=expected_amplitude,
        pixel_counts=sample_B[mask],
        pixel_vectors=xyz[:, mask]
    )
    prior_B = DipolePriorJax(
        mean_count_range=[0.9 * np.nanmean(sample_B), 1.1 * np.nanmean(sample_B)],
        speed_range=[0, 20], # [0, 20]
    )
    prior_B.rename_short_name('N', 'Nbar')
    prior_B.change_kwarg('Nbar', 'mean_density')

    if CATWISE_VERSION == 'S21':
        x0, mask = model.make_real_sample()
    elif CATWISE_VERSION == 'S22':
        x0 = np.asarray(
            np.load('dipolesbi/catwise/catwise_S22.npy'), dtype=np.float32
        )
        mask = model.binary_mask
        x0[~mask] = np.nan

        x0, mask = downgrade_ignore_nan(x0, mask, DOWNSCALE_NSIDE)

    os.makedirs(f'joint_logZ/{JOINT_SAMPLE}', exist_ok=True)
    out = run_ns_from_chkpt(
        path_to_chkpt=PATH_TO_CHKPT,
        data=x0,
        mask=mask,
        jax_key=jax.random.PRNGKey(PRNG_SEED),
        lnlike_B=lnlike_B,
        prior_B=prior_B,
        data_B=sample_B,
        save_dir=f'joint_logZ/{JOINT_SAMPLE}'
    )
    lnZ = out.logZ()
    lnZ_std = out.logZ(100).std()

    with open(f'joint_logZ/{JOINT_SAMPLE}/evidence.txt', 'w') as f:
        f.write(
            f'lnZ: {lnZ}\nlnZ_err: {lnZ_std}'
        )
