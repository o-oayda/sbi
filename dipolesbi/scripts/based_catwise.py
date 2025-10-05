from functools import partial
from typing import Optional
from numpy.typing import NDArray
from dipolesbi.catwise.maps import Catwise
from dipolesbi.tools.configs import CatwiseConfig, DataTransformSpec, Scenario
from dipolesbi.tools.multiround_inferer import MultiRoundInferer
from dipolesbi.tools.np_rngkey import NPKey
from dipolesbi.tools.priors_np import DipolePriorNP
from dipolesbi.tools.ui import MultiRoundInfererUI
import argparse
import numpy as np
from dipolesbi.tools.utils import batch_simulate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        help='Comma separated list of modes to run (e.g. "NLE" or "NLE,NPE").'
    )
    parser.add_argument(
        '--n_simulations',
        type=int,
        help='Number of simulations to run.'
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        help='Number of workers to distribute simuation over.'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        help='Name of directory in simulations to save into (automatically created).'
    )
    parser.add_argument(
        '--ssnle_seed',
        type=int,
        default=0,
        help='Seed used for sequential neural estimators.'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Choose a model to run.'
    )
    parser.add_argument(
        '--downscale_nside',
        type=int,
        default=None,
        help='Optional HEALPix nside to downscale simulated maps to.'
    )
    parser.add_argument(
        '--no_ui',
        action='store_true',
        help='Disable the Rich multi-round progress UI.'
    )
    args = parser.parse_args()

    MODE = args.mode
    N_SIM = args.n_simulations
    N_WORKERS = args.n_workers
    SAVE_DIR = args.out_dir
    USE_FLOAT32 = False
    N_ROUNDS = 10
    DOWNSCALE_NSIDE = args.downscale_nside

    def simulator_wrapper(**kwargs) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        return model.generate_dipole(**kwargs)

    prior = DipolePriorNP(
        mean_count_range=[np.log10(30_000_000), np.log10(40_000_000)],
        speed_range=[0, 8]
    )
    prior.change_kwarg(
        param_short_name='N',
        new_kwarg='log10_n_initial_samples'
    )

    def add_error_scale(short_name: str, prior: DipolePriorNP):
        prior.add_prior(
            short_name=short_name,
            simulator_kwarg='w1_extra_error',
            low=0,
            high=8,
            dist_type='uniform',
            index=1
        )

    def add_tdist_shape_param(prior: DipolePriorNP):
        prior.add_prior(
            short_name='nu',
            simulator_kwarg='log10_magnitude_error_shape_param',
            low=-1,
            high=3,
            dist_type='uniform',
            index=3
        )

    simulator = simulator_wrapper

    theta_0 = { # add a reference theta for diagnosing learned P(D | theta_0)
        'log10_n_initial_samples': 7.552,
        'w1_extra_error': 4.,
        'w2_extra_error': 3.2,
        'observer_speed': 2.,
        'dipole_longitude': 220,
        'dipole_latitude': 45
    }

    match args.model:
        case 'free_gauss_extra_err':
            ERROR_DIST = 'gaussian'
            COMMON_ERROR = True
            add_error_scale('etaWX', prior)
            theta_0.pop('w2_extra_error')

        case 'free_students-t_extra_err':
            ERROR_DIST = 'students-t'
            COMMON_ERROR = True
            add_error_scale('etaWX', prior)
            add_tdist_shape_param(prior)
            simulator = simulator_wrapper

        case 'free_gauss':
            ERROR_DIST = 'gaussian'
            COMMON_ERROR = None
            simulator = partial( # auto-add kwargs on call of simulator()
                simulator_wrapper,
                w1_extra_error=None,
                w2_extra_error=None
            )

        case 'cmb_dipole':
            ERROR_DIST = 'gaussian'
            COMMON_ERROR = True
            # the default params for the dipole are the CMB ones
            simulator = simulator_wrapper
            prior.remove_prior('D')
            prior.remove_prior('phi')
            prior.remove_prior('theta')
            add_error_scale('etaWX', prior)
            assert prior.ndim == 2

        case 'cmb_direction':
            ERROR_DIST = 'gaussian'
            COMMON_ERROR = True
            simulator = simulator_wrapper
            prior.remove_prior('phi')
            prior.remove_prior('theta')
            add_error_scale('etaWX', prior)
            assert prior.ndim == 3

        case 'cmb_velocity':
            ERROR_DIST = 'gaussian'
            COMMON_ERROR = True
            simulator = simulator_wrapper
            prior.remove_prior('D')
            add_error_scale('etaWX', prior)
            assert prior.ndim == 4

        case _:
            raise KeyError(f'Model {args.model} not recognised.')

    config = CatwiseConfig(
        cat_w1_max=17.0, 
        cat_w12_min=0.5,
        magnitude_error_dist=ERROR_DIST,
        use_float32=USE_FLOAT32,
        use_common_extra_error=COMMON_ERROR,
        model_identifier=args.model,
        downscale_nside=DOWNSCALE_NSIDE
    )

    model = Catwise(config)
    model.initialise_data()
    prior_jax = prior.to_jax()

    match args.mode:
        case 'NPE':
            scenario = Scenario.anynside_npe(
                nside=DOWNSCALE_NSIDE,
                theta_prior=prior_jax,
                reference_theta=theta_0,
                theta_spec_overrides={'embed_transform_in_flow': True},
                multiround_overrides={
                    'prng_integer_seed': args.ssnle_seed,
                    'plot_save_dir': SAVE_DIR,
                    'n_rounds': N_ROUNDS,
                    'simulation_budget': N_SIM
                },
                training_overrides={'learning_rate': 0.001}
            )
        case 'NLE':
            data_spec = DataTransformSpec.zscore(
                method='batchwise'
            )
            scenario = Scenario.anynside_nle(
                nside=DOWNSCALE_NSIDE,
                theta_prior=prior_jax,
                training_overrides={
                    'learning_rate': 1e-4,
                    'min_lr_ratio': 1.
                },
                reference_theta=theta_0,
                multiround_overrides={
                    'prng_integer_seed': args.ssnle_seed,
                    'plot_save_dir': SAVE_DIR,
                    'simulation_budget': N_SIM,
                    'n_rounds': N_ROUNDS,
                    'likelihood_chunk_size_gb': 0.5,
                    'n_likelihood_samples':  10_000
                },
                flow_overrides={
                    'decoder_n_neurons': 128,
                    'decoder_n_layers': 4,
                    'architecture': 4 * ['MAF'] + ['surjective_MAF'] + 6 * ['MAF'],
                    'data_reduction_factor': 0.5,
                },
                data_spec=data_spec
            )
        case _:
            raise KeyError(f'Mode {args.mode} not recognised.')


    def model_sim_wrapper(
            npkey: NPKey,
            params: dict[str, NDArray],
            noise: bool = True,
            ui: Optional[MultiRoundInfererUI] = None
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        return batch_simulate(
            params,
            simulator,
            n_workers=N_WORKERS,
            ui=ui
        )

    x0, mask = model.make_real_sample()
        
    inferer = MultiRoundInferer(
        MODE, prior, model_sim_wrapper, (x0, mask),
        multi_round_config=scenario.multiround,
        transform_config=scenario.transforms,
        nflow_config=scenario.flow,
        train_config=scenario.training,
        use_ui=not args.no_ui,
        model_config=config
    )
    inferer.run()
