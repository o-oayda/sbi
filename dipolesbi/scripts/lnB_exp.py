import argparse
import json
import os
import healpy as hp
import jax
from jax.random import PRNGKey
import numpy as np
from jax import numpy as jnp
from numpy.typing import ArrayLike
from typing import Callable
from dipolesbi.tools.configs import DataTransformSpec, Scenario, SimpleDipoleMapConfig
from dipolesbi.tools.constants import CMB_B, CMB_L
from dipolesbi.tools.jax_ns import JaxNestedSampler
from dipolesbi.tools.maps import SimpleDipoleMap, SimpleDipoleMapJax
from dipolesbi.tools.multiround_inferer import MultiRoundInferer
from dipolesbi.tools.np_rngkey import npkey_from_jax
from dipolesbi.tools.priors_np import DipolePriorNP


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nside',
        type=int,
        required=True,
        help='Nside of simulated maps.'
    )
    parser.add_argument(
        '--downscale_nside',
        type=int,
        required=True,
        help='Nside of downscaled maps to use in normalising flow.'
    )
    parser.add_argument(
        '--ssnle_seed',
        type=int,
        required=True,
        help='Integer seed for the multiround inferer pipeline.'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='Diretory for simulation outputs.'
    )
    args = parser.parse_args()

    x0_rng_key = PRNGKey(0)
    x0_rng_key_np = npkey_from_jax(x0_rng_key)
    NSIDE = args.nside 
    COARSE_NSIDE = args.downscale_nside
    TOTAL_SOURCES = 2_500_000
    MEAN_DENSITY = np.asarray(TOTAL_SOURCES / hp.nside2npix(NSIDE))
    MODELS = ['free_dipole', 'cmb_dipole', 'cmb_velocity', 'cmb_direction']
    theta0 = {
        'mean_density': MEAN_DENSITY,
        'observer_speed': np.asarray(1.5),
        'dipole_longitude': np.asarray(230.),
        'dipole_latitude': np.asarray(40.)
    }
    assert hp.isnsideok(NSIDE)
    assert hp.isnsideok(COARSE_NSIDE)

    # make a number of pseudo-CatSIMs, compute the analytic lnBs at high nside,
    # compare with lnBs at low nside
    simpledipole_config = SimpleDipoleMapConfig(
        nside=NSIDE, downscale_nside=COARSE_NSIDE
    )
    model = SimpleDipoleMap(simpledipole_config)
    model.catwise_mask()
    x0, coarse_mask = model.generate_dipole(x0_rng_key_np, theta=theta0)
    native_dmap, native_mask = model.dmap_and_mask
    true_model_jax = SimpleDipoleMapJax( # yeah this is jank but deal with it
        nside=NSIDE,
        downscale_nside=COARSE_NSIDE,
        reference_data=jax.device_put(native_dmap).squeeze(),
        reference_mask=jax.device_put(native_mask).squeeze()
    )

    assert native_mask.shape[-1] == 12 * NSIDE ** 2
    assert native_dmap.shape[-1] == 12 * NSIDE ** 2
    assert coarse_mask.shape[-1] == 12 * COARSE_NSIDE ** 2
    assert x0.shape[-1]          == 12 * COARSE_NSIDE ** 2

    prior = DipolePriorNP(
        mean_count_range=[float(0.95*MEAN_DENSITY), float(1.05*MEAN_DENSITY)],
    )
    prior.change_kwarg('N', 'mean_density')
    prior_jax = prior.to_jax()
    adapter = prior_jax.get_adapter()

    data_spec = DataTransformSpec.zscore(
        method='batchwise'
    )
    cur_cfg = Scenario.anynside_nle(
        nside=COARSE_NSIDE, # since the flow only sees the coarse res
        reference_theta=theta0, # type: ignore
        theta_prior=prior_jax,
        training_overrides={
            'learning_rate': 1e-4,
            'min_lr_ratio': 1.
        },
        multiround_overrides={
            'prng_integer_seed': args.ssnle_seed,
            'plot_save_dir': args.out_dir,
            'simulation_budget': 50_000,
            'n_rounds': 15,
            'likelihood_chunk_size_gb': 0.5,
            'n_likelihood_samples': 5_000
        },
        flow_overrides={
            'decoder_n_neurons': 128,
            'decoder_n_layers': 4,
            'architecture': 4 * ['MAF'] + ['surjective_MAF'] + 6 * ['MAF'],
            'data_reduction_factor': 0.5,
        },
        data_spec=data_spec
    )

    # do inference on the coarse data to obtain the coarse lnlike
    # don't worry about true logl --- we'll do the benchmarking manually
    inferer = MultiRoundInferer(
        'NLE', prior, model.generate_dipole, (x0, coarse_mask),
        multi_round_config=cur_cfg.multiround,
        transform_config=cur_cfg.transforms,
        nflow_config=cur_cfg.flow,
        train_config=cur_cfg.training,
        use_ui=False,
        model_config=simpledipole_config
    )
    inferer.run()

    # from the plumbing of the multiround inferer
    mask0 = np.asarray(inferer.reference_mask, dtype=bool)
    mask0 = np.broadcast_to(mask0, np.asarray(x0).shape)
    z0_np, zmask0_np, log_det_jac_np = inferer._transform_data_and_logdet(
        np.asarray(x0),
        mask0
    )
    z0 = jax.device_put(z0_np)
    zmask0 = jax.device_put(zmask0_np)
    log_det_jac = jax.device_put(log_det_jac_np)

    def NLE_lnlike(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        assert inferer.theta_transform is not None
        theta, _ = inferer.theta_transform(params, in_ns=True)

        assert inferer.nflow is not None
        log_like = inferer.nflow.evaluate_lnlike(
            theta[None, :], 
            z0, 
            mask=zmask0
        )

        log_like += log_det_jac
        return log_like.squeeze()

    def NLE_lnlike_wrapper(theta: dict[str, jnp.ndarray]) -> jnp.ndarray:
        return NLE_lnlike(theta)

    def true_logl_wrapper(theta: dict[str, jnp.ndarray]) -> jnp.ndarray:
        return true_model_jax.log_likelihood(theta)

    def with_fixed_kwargs(
            base_fn: Callable[[dict[str, jnp.ndarray]], jnp.ndarray],
            **fixed_kwargs: ArrayLike
    ):
        array_fixed_kwargs = {
            key: jnp.asarray(val) for key, val in fixed_kwargs.items()
        }

        def wrapped(theta: dict[str, jnp.ndarray]) -> jnp.ndarray:
            theta_with_fixed = dict(theta)
            theta_with_fixed.update(array_fixed_kwargs)
            return base_fn(theta_with_fixed)

        return wrapped

    # do analytic, high-nside NS for a number of models and store
    current_key = x0_rng_key
    lnZ_results = {}
    for model in MODELS:
        ns_key, current_key = jax.random.split(current_key)
        prior = DipolePriorNP(
            mean_count_range=[float(0.95*MEAN_DENSITY), float(1.05*MEAN_DENSITY)],
        )
        prior.change_kwarg('N', 'mean_density')

        match model:
            case 'free_dipole':
                true_logl = true_logl_wrapper
                NLE_logl = NLE_lnlike_wrapper
            case 'cmb_dipole':
                prior.remove_prior('D')
                prior.remove_prior('phi')
                prior.remove_prior('theta')
                true_logl = with_fixed_kwargs(
                    true_logl_wrapper,
                    observer_speed=1,
                    dipole_longitude=CMB_L,
                    dipole_latitude=CMB_B
                )
                NLE_logl = with_fixed_kwargs(
                    NLE_lnlike_wrapper,
                    observer_speed=1,
                    dipole_longitude=CMB_L,
                    dipole_latitude=CMB_B
                )
            case 'cmb_velocity':
                prior.remove_prior('D')
                true_logl = with_fixed_kwargs(
                    true_logl_wrapper,
                    observer_speed=1
                )
                NLE_logl = with_fixed_kwargs(
                    NLE_lnlike_wrapper,
                    observer_speed=1
                )
            case 'cmb_direction':
                prior.remove_prior('phi')
                prior.remove_prior('theta')
                true_logl = with_fixed_kwargs(
                    true_logl_wrapper,
                    dipole_longitude=CMB_L,
                    dipole_latitude=CMB_B
                )
                NLE_logl = with_fixed_kwargs(
                    NLE_lnlike_wrapper,
                    dipole_longitude=CMB_L,
                    dipole_latitude=CMB_B
                )
            case _:
                raise ValueError(f'Model ({model}) not recognised.')

        prior_jax = prior.to_jax()
        jax_ns = JaxNestedSampler(true_logl, prior_jax)
        jax_ns.setup(ns_key, n_live=1000, n_delete=200)
        results = jax_ns.run()

        true_lnZ = float(results.logZ()) # type: ignore
        true_lnZerr = float(results.logZ(100).std()) # type: ignore
        
        lnZ_results.setdefault(model, {}).setdefault('true', {})
        lnZ_results[model]['true']['lnZ'] = true_lnZ
        lnZ_results[model]['true']['err'] = true_lnZerr

        ns_key, current_key = jax.random.split(current_key)
        jax_ns_NLE = JaxNestedSampler(NLE_logl, prior_jax)
        jax_ns_NLE.setup(ns_key, n_live=1000, n_delete=200)
        results = jax_ns_NLE.run()

        NLE_lnZ = float(results.logZ()) # type: ignore
        NLE_lnZerr = float(results.logZ(100).std()) # type: ignore
        
        lnZ_results.setdefault(model, {}).setdefault('nle', {})
        lnZ_results[model]['nle']['lnZ'] = NLE_lnZ
        lnZ_results[model]['nle']['err'] = NLE_lnZerr

    save_dir = inferer.mr_config.plot_save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'lnZ_results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(lnZ_results, f, indent=2)
