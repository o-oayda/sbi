from contextlib import nullcontext
from functools import partial
from typing import Optional
from catsim.utils.healsphere import downgrade_ignore_nan
from numpy.typing import NDArray
from catsim import Catwise, CatwiseConfig
from dipolesbi.tools.configs import DataTransformSpec, Scenario
from dipolesbi.tools.multiround_inferer import MultiRoundInferer
from dipolesbi.tools.np_rngkey import NPKey
from dipolesbi.tools.priors_np import DipolePriorNP
from dipolesbi.tools.ui import MultiRoundInfererUI
import argparse
from dataclasses import asdict
import numpy as np
from joblib import parallel_backend
from dipolesbi.tools.utils import batch_simulate
from dipolesbi.tools.remote_sim import ensure_worker_model, remote_generate_dipole
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        nargs='+',
        help='One or more modes to run, separated by spaces (e.g. "--mode NLE NPE").'
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
        help=(
            'Optional HEALPix nside to downscale simulated maps to. '
            'Ignored for NPE runs, which always operate on the native nside (64).'
        )
    )
    parser.add_argument(
        '--no_ui',
        action='store_true',
        help='Disable the Rich multi-round progress UI.'
    )
    parser.add_argument(
        '--simulation_backend',
        choices=['joblib', 'dask'],
        default='joblib',
        help='Choose how simulations are dispatched; defaults to local joblib workers.'
    )
    parser.add_argument(
        '--dask_scheduler',
        type=str,
        default=None,
        help='Optional Dask scheduler address (e.g. "tcp://head:8786") when using --simulation_backend dask.'
    )
    parser.add_argument(
        '--expected_dask_workers',
        type=int,
        default=None,
        help='Optional hint for how many Dask workers to wait for before starting simulations.'
    )
    parser.add_argument(
        '--catwise_version',
        choices=['S21', 'S22'],
        default='S21',
        help='Specify whether to use the Secrest+21 or Secrest+22 sample for inference.'
    )
    parser.add_argument(
        '--use_clusters',
        action='store_true',
        help='If specified, generated clustered/correlated points.'
    )
    args = parser.parse_args()

    raw_modes = args.mode or []
    modes: list[str] = []
    for entry in raw_modes:
        modes.extend(part.strip().upper() for part in entry.split(',') if part.strip())
    if not modes:
        parser.error('At least one mode must be provided via --mode.')

    N_SIM = args.n_simulations
    N_WORKERS = args.n_workers
    SAVE_DIR = args.out_dir
    USE_FLOAT32 = False
    N_ROUNDS = 15
    DOWNSCALE_NSIDE = args.downscale_nside
    NPE_DOWNSCALE_NSIDE = 32
    ORIGINAL_NSIDE = 64
    USE_CLUSTERS = args.use_clusters
    SIM_BACKEND = args.simulation_backend
    DASK_SCHEDULER = args.dask_scheduler
    dask_client = None
    warmed_configs = set()
    EXPECTED_DASK_WORKERS = (
        args.expected_dask_workers if args.expected_dask_workers and args.expected_dask_workers > 0 else None
    )

    if SIM_BACKEND != 'dask' and DASK_SCHEDULER is not None:
        parser.error(
            '--dask_scheduler is only valid when using --simulation_backend dask.'
        )
    if SIM_BACKEND == 'dask':
        try:
            from dask.distributed import Client, get_client
        except ImportError as exc:
            parser.error(
                'Using --simulation_backend dask requires dask.distributed to be installed.'
            )
        try:
            dask_client = get_client()
        except ValueError:
            dask_client = Client(DASK_SCHEDULER) if DASK_SCHEDULER else Client()
        try:
            target_workers = EXPECTED_DASK_WORKERS or 1
            dask_client.wait_for_workers(target_workers, timeout=120)
        except TimeoutError:
            print(
                'Warning: Timed out waiting for the expected number of Dask workers; '
                'continuing anyway.'
            )

    def _parallel_backend_context():
        if SIM_BACKEND == 'dask':
            backend_kwargs = {}
            if DASK_SCHEDULER:
                backend_kwargs['scheduler_host'] = DASK_SCHEDULER
            return parallel_backend('dask', **backend_kwargs)
        return nullcontext()

    def simulator_wrapper(
            rng_key: Optional[NPKey] = None,
            **kwargs
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        return model.generate_dipole(
            rng_key=rng_key,
            w1_max=16.5 if args.catwise_version == 'S22' else 16.4,
            **kwargs
        )

    prior = DipolePriorNP(
        mean_count_range=[np.log10(30_000_000), np.log10(40_000_000)], #U[7.5,7.6]
        speed_range=[0, 8] # [0, 20]
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
            high=8, # U[0,5]
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

    def add_cluster_params(prior: DipolePriorNP):
        prior.add_prior(
            short_name='l_clus',
            simulator_kwarg='cluster_rate_param',
            low=5,
            high=100,
            dist_type='uniform',
        )
        prior.add_prior(
            short_name='kappa',
            simulator_kwarg='log10_cluster_scale_param',
            low=3,
            high=5,
            dist_type='uniform',
        )

    simulator = simulator_wrapper

    theta_0 = { # add a reference theta for diagnosing learned P(D | theta_0)
        'log10_n_initial_samples': 7.552,
        'w1_extra_error': 4.,
        # 'w2_extra_error': 3.2,
        'observer_speed': 2.,
        'dipole_longitude': 220,
        'dipole_latitude': 45
    }

    if USE_CLUSTERS:
        add_cluster_params(prior)
        theta_0['cluster_rate_param'] = 10
        theta_0['log10_cluster_scale_param'] = 3

    match args.model:
        case 'free_gauss_extra_err':
            ERROR_DIST = 'gaussian'
            COMMON_ERROR = True
            add_error_scale('etaWX', prior)
            # theta_0.pop('w2_extra_error')

        case 'free_students-t_extra_err':
            ERROR_DIST = 'students-t'
            COMMON_ERROR = True
            add_error_scale('etaWX', prior)
            add_tdist_shape_param(prior)
            simulator = simulator_wrapper
            theta_0['log10_magnitude_error_shape_param'] = 1.

        case 'free_gauss':
            ERROR_DIST = 'gaussian'
            COMMON_ERROR = None
            simulator = partial( # auto-add kwargs on call of simulator()
                simulator_wrapper,
                w1_extra_error=None,
                w2_extra_error=None
            )
            theta_0.pop('w1_extra_error')

        case 'free_students-t':
            ERROR_DIST = 'students-t'
            COMMON_ERROR = None
            add_tdist_shape_param(prior)
            simulator = partial(
                simulator_wrapper,
                w1_extra_error=None,
                w2_extra_error=None
            )
            theta_0.pop('w1_extra_error')
            theta_0['log10_magnitude_error_shape_param'] = 1.

        case 'cmb_dipole':
            ERROR_DIST = 'gaussian'
            COMMON_ERROR = True
            # the default params for the dipole are the CMB ones
            simulator = simulator_wrapper
            prior.remove_prior('D')
            prior.remove_prior('phi')
            prior.remove_prior('theta')
            add_error_scale('etaWX', prior)
            theta_0.pop('observer_speed')
            theta_0.pop('dipole_longitude')
            theta_0.pop('dipole_latitude')
            assert prior.ndim == 2

        case 'cmb_direction':
            ERROR_DIST = 'gaussian'
            COMMON_ERROR = True
            simulator = simulator_wrapper
            prior.remove_prior('phi')
            prior.remove_prior('theta')
            add_error_scale('etaWX', prior)
            theta_0.pop('dipole_longitude'); theta_0.pop('dipole_latitude')
            assert prior.ndim == 3

        case 'cmb_velocity':
            ERROR_DIST = 'gaussian'
            COMMON_ERROR = True
            simulator = simulator_wrapper
            prior.remove_prior('D')
            add_error_scale('etaWX', prior)
            assert prior.ndim == 4
            theta_0.pop('observer_speed')

        case 'secrest+21':
            ERROR_DIST = 'gaussian'
            COMMON_ERROR = True
            simulator = partial(
                simulator_wrapper,
                observer_speed=2.156, # from CatWISE_Dipole_results.ipynb in secrest's code
                dipole_longitude=238.2,
                dipole_latitude=28.8
            )
            prior.remove_prior('D')
            prior.remove_prior('phi')
            prior.remove_prior('theta')
            add_error_scale('etaWX', prior)
            theta_0.pop('observer_speed')
            theta_0.pop('dipole_longitude')
            theta_0.pop('dipole_latitude')
            assert prior.ndim == 2

        case 'dam+23':
            ERROR_DIST = 'gaussian'
            COMMON_ERROR = True
            simulator = partial(
                simulator_wrapper,
                observer_speed=2.68, # from CatWISE_Dipole_results.ipynb in secrest's code
                dipole_longitude=237.2,
                dipole_latitude=41.8
            )
            prior.remove_prior('D')
            prior.remove_prior('phi')
            prior.remove_prior('theta')
            add_error_scale('etaWX', prior)
            theta_0.pop('observer_speed')
            theta_0.pop('dipole_longitude')
            theta_0.pop('dipole_latitude')
            assert prior.ndim == 2

        case _:
            raise KeyError(f'Model {args.model} not recognised.')


    prior_jax = prior.to_jax()

    for mode in modes:
        match mode:
            case 'NPE':
                nside = NPE_DOWNSCALE_NSIDE
                current_downscale = NPE_DOWNSCALE_NSIDE
                # if DOWNSCALE_NSIDE not in (None, ORIGINAL_NSIDE):
                print(
                    f'Using hardcoded {NPE_DOWNSCALE_NSIDE} for NPE downscale nside.'
                    # f'Overriding --downscale_nside={DOWNSCALE_NSIDE} to native '
                    # f'nside={ORIGINAL_NSIDE} for NPE.'
                )
                scenario = Scenario.anynside_npe(
                    nside=nside,
                    theta_prior=prior_jax,
                    reference_theta=theta_0,
                    theta_spec_overrides={'embed_transform_in_flow': True},
                    multiround_overrides={
                        'prng_integer_seed': args.ssnle_seed,
                        'plot_save_dir': SAVE_DIR,
                        'n_rounds': N_ROUNDS,
                        'simulation_budget': N_SIM,
                        'likelihood_chunk_size_gb': 0.5,
                        'n_likelihood_samples':  10_000
                    },
                    training_overrides={'learning_rate': 0.001}
                )
            case 'NLE':
                nside = DOWNSCALE_NSIDE
                current_downscale = DOWNSCALE_NSIDE
                data_spec = DataTransformSpec.zscore(
                    method='batchwise'
                )
                scenario = Scenario.anynside_nle(
                    nside=nside, # type: ignore
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
                raise KeyError(f'Mode {mode} not recognised.')

        config = CatwiseConfig(
            cat_w1_max=17.0, 
            cat_w12_min=0.5,
            magnitude_error_dist=ERROR_DIST,
            use_float32=USE_FLOAT32,
            use_common_extra_error=COMMON_ERROR,
            model_identifier=args.model,
            downscale_nside=current_downscale,
            base_mask_version=args.catwise_version,
            generate_correlated_points=USE_CLUSTERS
        )

        model = Catwise(config)
        model.initialise_data()

        if SIM_BACKEND == 'dask':
            config_payload = asdict(config)
            if dask_client is not None:
                config_key = tuple(sorted(config_payload.items()))
                if config_key not in warmed_configs:
                    try:
                        dask_client.run(ensure_worker_model, config_payload)
                        warmed_configs.add(config_key)
                    except OSError as exc:
                        print(
                            'Warning: Failed to warm Dask workers for Catwise config; '
                            f'continuing without preloading ({exc}).'
                        )

            def _make_remote_sim_callable(base_callable):
                fixed_kwargs: dict[str, object] = {}
                current = base_callable
                while isinstance(current, partial):
                    if current.keywords:
                        fixed_kwargs = {**current.keywords, **fixed_kwargs}
                    current = current.func

                def _remote_sim_callable(**kwargs):
                    call_kwargs = {**fixed_kwargs, **kwargs}
                    rng_key = call_kwargs.pop('rng_key', None)
                    return remote_generate_dipole(config_payload, call_kwargs, rng_key)

                return _remote_sim_callable

            sim_callable = _make_remote_sim_callable(simulator)
            parallel_opts = {'batch_size': 1}
        else:
            sim_callable = simulator
            parallel_opts = None

        def model_sim_wrapper(
                npkey: NPKey,
                params: dict[str, NDArray],
                noise: bool = True,
                ui: Optional[MultiRoundInfererUI] = None
        ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
            with _parallel_backend_context():
                return batch_simulate(
                    params,
                    sim_callable,
                    n_workers=N_WORKERS,
                    ui=ui,
                    rng_key=npkey,
                    parallel_kwargs=parallel_opts
                )

        if args.catwise_version == 'S21':
            x0, mask = model.make_real_sample()
        elif args.catwise_version == 'S22':
            x0 = np.asarray(
                np.load('dipolesbi/catwise/catwise_S22.npy'), dtype=np.float32
            )
            mask = model.binary_mask
            x0[~mask] = np.nan

            if DOWNSCALE_NSIDE:
                x0, mask = downgrade_ignore_nan(x0, mask, DOWNSCALE_NSIDE)
        else:
            raise ValueError(
                f'Catwise version ({args.catwise_version} not recognised).'
            )

        inferer = MultiRoundInferer(
            mode, prior, model_sim_wrapper, (x0, mask),
            multi_round_config=scenario.multiround,
            transform_config=scenario.transforms,
            nflow_config=scenario.flow,
            train_config=scenario.training,
            use_ui=not args.no_ui,
            model_config=config
        )
        inferer.run()
