from typing import Optional
from numpy.typing import NDArray
from dipolesbi.catwise.maps import Catwise
from dipolesbi.tools.configs import Scenario
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
        '--error_dist',
        type=str,
        default='gaussian',
        help='Error dist. for CatWISE errors (gaussian or students-t; default: gaussian).'
    )
    args = parser.parse_args()

    MODE = args.mode
    N_SIM = args.n_simulations
    N_WORKERS = args.n_workers
    SAVE_DIR = args.out_dir
    ERROR_DIST = args.error_dist
    USE_FLOAT32 = False
    NSIDE = 64

    model = Catwise(
        cat_w1_max=17.0, 
        cat_w12_min=0.5,
        magnitude_error_dist=ERROR_DIST,
        use_float32=USE_FLOAT32
    )
    model.initialise_data()
    prior = DipolePriorNP(
        mean_count_range=[30_000_000, 40_000_000],
        speed_range=[0, 8]
    )
    prior.add_prior(
        short_name='etaW1',
        simulator_kwarg='w1_extra_error',
        low=0,
        high=8,
        dist_type='uniform',
        index=1
    )
    prior.add_prior(
        short_name='etaW2',
        simulator_kwarg='w2_extra_error',
        low=0,
        high=8,
        dist_type='uniform',
        index=2
    )
    if ERROR_DIST == 'students-t':
        prior.add_prior(
            short_name='nu',
            simulator_kwarg='log10_magnitude_error_shape_param',
            low=-1,
            high=3,
            dist_type='uniform',
            index=3
        )

    prior_jax = prior.to_jax()

    nside16_scenario_npe = Scenario.anynside_npe(
        nside=NSIDE,
        theta_prior=prior_jax,
        theta_spec_overrides={'embed_transform_in_flow': True},
        multiround_overrides={
            'prng_integer_seed': args.ssnle_seed,
            'plot_save_dir': SAVE_DIR,
            'n_rounds': 10,
            'check_proposal_probs': True,
            'simulation_budget': N_SIM
        },
        training_overrides={'learning_rate': 0.001}
    )
    nside16_scenario_nle = Scenario.anynside_nle(
        nside=NSIDE,
        theta_prior=prior_jax,
        multiround_overrides={
            'prng_integer_seed': args.ssnle_seed,
            'plot_save_dir': SAVE_DIR,
            'simulation_budget': 100_000,
            'n_rounds': 20
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
    cur_cfg = meta_cfg[MODE]

    def model_sim_wrapper(
            npkey: NPKey,
            params: dict[str, NDArray],
            noise: bool = True,
            ui: Optional[MultiRoundInfererUI] = None
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        return batch_simulate(
            params,
            model.generate_dipole,
            n_workers=N_WORKERS,
            ui=ui
        )

    x0, mask = model.make_real_sample()
        
    inferer = MultiRoundInferer(
        MODE, prior, model_sim_wrapper, (x0, mask),
        multi_round_config=cur_cfg.multiround,
        transform_config=cur_cfg.transforms,
        nflow_config=cur_cfg.flow,
        train_config=cur_cfg.training
    )
    inferer.run()
