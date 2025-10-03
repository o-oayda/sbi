from jax.random import PRNGKey
from dipolesbi.tools.healpix_helpers import downgrade_ignore_nan
from dipolesbi.tools.multiround_inferer import MultiRoundInferer
from dipolesbi.tools.configs import Scenario
from dipolesbi.tools.np_rngkey import npkey_from_jax
from dipolesbi.tools.maps import SimpleDipoleMap, SimpleDipoleMapJax
import healpy as hp
import numpy as np
from dipolesbi.tools.priors_np import DipolePriorNP
import matplotlib.pyplot as plt
import argparse
import jax


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nside',
        type=int,
        help='Nside of simulated maps.'
    )
    parser.add_argument(
        '--mode',
        type=str,
        help='Comma separated list of modes to run (e.g. "NLE" or "NLE,NPE").'
    )
    parser.add_argument(
        '--ssnle_seed',
        type=int,
        help='Integer seed for the multiround inferer pipeline.'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        help='Diretory for simulation outputs.'
    )
    args = parser.parse_args()

    x0_rng_key = PRNGKey(42)

    NSIDE = args.nside 
    COARSE_NSIDE = NSIDE // 2
    TOTAL_SOURCES = 1_920_000
    MEAN_DENSITY = np.asarray(TOTAL_SOURCES / hp.nside2npix(NSIDE))
    modes = []
    if args.mode is not None:
        for raw_mode in args.mode.split(','):
            cleaned = raw_mode.strip()
            if cleaned:
                modes.append(cleaned.upper())
    if not modes:
        raise ValueError('Provide at least one mode via --mode.')
    theta0 = {
        'mean_density': MEAN_DENSITY,
        'observer_speed': np.asarray(2.),
        'dipole_longitude': np.asarray(215.),
        'dipole_latitude': np.asarray(40.)
    }

    model = SimpleDipoleMap(nside=NSIDE, downscale_nside=COARSE_NSIDE)
    model.equatorial_plane_mask(angle=30)
    # these will be at the coarse resolution
    x0, coarse_mask = model.generate_dipole(npkey_from_jax(x0_rng_key), theta=theta0)
    _, native_mask = model.dmap_and_mask

    # check downgrading working as intended
    assert native_mask.shape[-1] == 12 * NSIDE * NSIDE
    assert coarse_mask.shape[-1] == 12 * COARSE_NSIDE * COARSE_NSIDE
    assert x0.shape[-1] == 12 * COARSE_NSIDE * COARSE_NSIDE

    # this will be a bit how you going until I can work out an explicit likelihood
    # function for the downgraded data --- in principle doable since we know
    # the likelihood of the constituent pixels and how they are used to form the
    # coarser map
    # x0, mask = downgrade_ignore_nan(x0, mask, nside_out=8)
    # def model_sim_wrapper(key, theta, make_poisson_draws):
    #     x, m = model.generate_dipole(key, theta, make_poisson_draws)
    #     x, m = downgrade_ignore_nan(x, m, nside_out=8)
    #     return x, m

    hp.projview(x0.squeeze() * coarse_mask.squeeze(), nest=True)
    plt.savefig('example_sample.pdf', bbox_inches='tight')
    plt.show()

    prior = DipolePriorNP(
        mean_count_range=[float(0.95*MEAN_DENSITY), float(1.05*MEAN_DENSITY)],
    )
    prior.change_kwarg('N', 'mean_density')
    prior_jax = prior.to_jax()
    adapter = prior_jax.get_adapter()

    true_model_jax = SimpleDipoleMapJax(
        nside=NSIDE,
        downscale_nside=COARSE_NSIDE,
        reference_data=jax.device_put(x0).squeeze(), # coarse resolution
        reference_mask=jax.device_put(native_mask).squeeze() # native resolution
    )

    nside16_scenario_npe = Scenario.anynside_npe(
        nside=COARSE_NSIDE, # since the flow only sees the coarse res
        reference_theta=theta0,
        theta_prior=prior_jax,
        theta_spec_overrides={'embed_transform_in_flow': True},
        multiround_overrides={
            'prng_integer_seed': args.ssnle_seed,
            'plot_save_dir': args.out_dir,
            'n_rounds': 10,
            'check_proposal_probs': True
        },
        training_overrides={'learning_rate': 0.001}
    )
    nside16_scenario_nle = Scenario.anynside_nle(
        nside=COARSE_NSIDE, # since the flow only sees the coarse res
        reference_theta=theta0,
        theta_prior=prior_jax,
        multiround_overrides={
            'prng_integer_seed': args.ssnle_seed,
            'plot_save_dir': args.out_dir,
            'simulation_budget': 100_000,
            'n_rounds': 20,
            'likelihood_chunk_size_gb': 0.5,
            'n_likelihood_samples': 5_000
        },
        flow_overrides={
            'decoder_n_neurons': 128,
            'decoder_n_layers': 4
        }
    )

    meta_cfg = {
        'NLE': nside16_scenario_nle,
        'NPE': nside16_scenario_npe
    }

    for MODE in modes:
        if MODE not in meta_cfg:
            raise ValueError(f"Mode '{MODE}' not recognised.")

        cur_cfg = meta_cfg[MODE]

        inferer = MultiRoundInferer(
            MODE, prior, model.generate_dipole, (x0, coarse_mask),
            true_logl=true_model_jax.log_likelihood,
            multi_round_config=cur_cfg.multiround,
            transform_config=cur_cfg.transforms,
            nflow_config=cur_cfg.flow,
            train_config=cur_cfg.training
        )
        inferer.run()
