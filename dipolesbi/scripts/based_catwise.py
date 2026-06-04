from functools import partial
from typing import Any, Optional
from catsim.utils.healsphere import downgrade_ignore_nan
from numpy.typing import NDArray
from catsim import Catwise, CatwiseConfig, CatwiseJax
from dipolesbi.tools.configs import DataTransformSpec, Scenario
from dipolesbi.tools.multiround_inferer import MultiRoundInferer
from dipolesbi.tools.np_rngkey import NPKey
from dipolesbi.tools.priors_np import DipolePriorNP
from dipolesbi.tools.ui import MultiRoundInfererUI
import argparse
import jax
import numpy as np
from dipolesbi.tools.utils import batch_simulate
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
        '--n_rounds',
        type=int,
        default=15,
        help='Specify the number of rounds of inference.'
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        help='Number of local workers to distribute NumPy simulation over.'
    )
    parser.add_argument(
        '--use_jax',
        action='store_true',
        help="Use catsim's JAX batched CatWISE simulator."
    )
    parser.add_argument(
        '--jax_batch_size',
        type=int,
        default=5,
        help='Batch size passed to CatwiseJax.batch_generate_dipole.'
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=None,
        help=(
            'Chunk size used inside the simulator. Defaults to 25_000 for '
            'NumPy and 140_000 for JAX.'
        )
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
        '--append_native_count_summary',
        action='store_true',
        help=(
            'Append the native-nside log count-dispersion summary statistic '
            'to the downscaled map for NLE.'
        )
    )
    parser.add_argument(
        '--no_ui',
        action='store_true',
        help='Disable the Rich multi-round progress UI.'
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
    parser.add_argument(
        '--simulate_clustering',
        type=str,
        choices=['poisson'],
        default=None,
        help='Clustering model to infer. Only "poisson" is supported.'
    )
    parser.add_argument(
        '--unique_error',
        action='store_true',
        help='If specified, make eta a unique term for W1 and W2 and not common.'
    )
    parser.add_argument(
        '--noecl_mask',
        action='store_true',
        help=(
            'If specified, use a catalogue version where the north ecliptic'
            'pole has not been masked.'
        )
    )
    parser.add_argument(
        '--add_confusion_noise',
        action='store_true'
    )
    parser.add_argument(
        '--wide_v_prior',
        action='store_true'
    )
    parser.add_argument(
        "--max_children",
        type=int,
        default=16
    )
    args = parser.parse_args()

    if args.chunk_size is None:
        args.chunk_size = 140_000 if args.use_jax else 25_000
    if args.use_jax and args.n_workers is not None:
        parser.error('--n_workers is only used by the NumPy simulator.')
    if args.use_jax and args.jax_batch_size <= 0:
        parser.error('--jax_batch_size must be positive.')
    if args.use_jax and args.use_clusters:
        parser.error('CatwiseJax does not yet support --use_clusters.')
    if args.use_jax and args.add_confusion_noise:
        parser.error('CatwiseJax does not yet support --add_confusion_noise.')
    if args.simulate_clustering == 'poisson' and args.use_clusters:
        parser.error('--simulate_clustering poisson cannot be combined with --use_clusters.')

    raw_modes = args.mode or []
    modes: list[str] = []
    for entry in raw_modes:
        modes.extend(part.strip().upper() for part in entry.split(',') if part.strip())
    if not modes:
        parser.error('At least one mode must be provided via --mode.')
    if args.append_native_count_summary and args.downscale_nside is None:
        parser.error('--append_native_count_summary requires --downscale_nside.')
    if args.append_native_count_summary and any(mode != 'NLE' for mode in modes):
        parser.error('--append_native_count_summary is currently supported for NLE only.')

    N_SIM = args.n_simulations
    N_WORKERS = args.n_workers
    SAVE_DIR = args.out_dir
    USE_FLOAT32 = False
    N_ROUNDS = args.n_rounds
    DOWNSCALE_NSIDE = args.downscale_nside
    NPE_DOWNSCALE_NSIDE = 32
    ORIGINAL_NSIDE = 64
    COMMON_ERROR = not args.unique_error
    NOECL_MASK = args.noecl_mask
    ADD_CONFUSION = args.add_confusion_noise
    WIDE_V_PRIOR = args.wide_v_prior
    USE_CLUSTERS = args.use_clusters
    APPEND_NATIVE_COUNT_SUMMARY = args.append_native_count_summary
    SIMULATE_CLUSTERING = args.simulate_clustering

    def simulator_wrapper(
            rng_key: Optional[NPKey] = None,
            **kwargs
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        return model.generate_dipole(
            rng_key=rng_key,
            w1_max=16.5 if args.catwise_version == 'S22' else 16.4,
            **kwargs
        )

    def _partial_keywords(func: Any) -> dict[str, Any]:
        fixed_kwargs: dict[str, Any] = {}
        current = func
        while isinstance(current, partial):
            if current.keywords:
                fixed_kwargs = {**current.keywords, **fixed_kwargs}
            current = current.func
        return fixed_kwargs

    def _jax_key_from_npkey(key: NPKey) -> jax.Array:
        if isinstance(key, NPKey):
            key_data = key._ss.generate_state(2, dtype=np.uint32)
        else:
            key_data = np.asarray(key, dtype=np.uint32).reshape(2)
        return jax.device_put(key_data)

    def _n_sims_from_params(params: dict[str, NDArray]) -> int:
        if 'log10_n_initial_samples' in params:
            values = np.asarray(params['log10_n_initial_samples'])
        else:
            values = np.asarray(next(iter(params.values())))
        return 1 if values.ndim == 0 else int(values.shape[0])

    def _batch_parameter(value: Any, n_sims: int) -> NDArray:
        if value is None:
            value = np.nan
        values = np.asarray(value)
        if values.ndim == 0:
            return np.full(n_sims, values.item())
        if values.shape != (n_sims,):
            raise ValueError(
                f'Fixed JAX simulator parameter must be scalar or shape ({n_sims},).'
            )
        return values

    def _native_counts(
            native_map: NDArray,
            native_mask: NDArray[np.bool_]
    ) -> NDArray[np.int64]:
        if native_map.shape != native_mask.shape:
            raise ValueError('native_map and native_mask must have matching shapes.')

        valid = native_mask.astype(bool, copy=False) & np.isfinite(native_map)
        if not np.any(valid):
            raise ValueError(
                'Cannot build native-count summary with no unmasked native pixels.'
            )

        counts = np.asarray(native_map[valid])
        rounded_counts = np.rint(counts)
        if np.any(rounded_counts < 0):
            raise ValueError('Native-count summary received negative counts.')
        return rounded_counts.astype(np.int64)

    def _native_count_log_dispersion_feature(
            native_map: NDArray,
            native_mask: NDArray[np.bool_]
    ) -> NDArray[np.float32]:
        counts = _native_counts(native_map, native_mask).astype(np.float64)
        if counts.size < 2:
            raise ValueError(
                'Native-count log dispersion requires at least two pixels.'
            )
        mean = float(np.mean(counts))
        if mean <= 0.0:
            raise ValueError(
                'Native-count log dispersion requires positive mean counts.'
            )
        dispersion = float(np.var(counts, ddof=1) / mean)
        if dispersion <= 0.0:
            raise ValueError(
                'Native-count log dispersion requires positive dispersion.'
            )
        return np.asarray([np.log(dispersion)], dtype=np.float32)

    def _build_hybrid_sample_from_native(
            native_map: NDArray,
            native_mask: NDArray[np.bool_],
            downscale_nside: int
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        coarse_map, coarse_mask = downgrade_ignore_nan(
            native_map,
            native_mask,
            downscale_nside,
        )
        coarse_map = coarse_map.astype(np.float32, copy=False)
        coarse_mask = coarse_mask.astype(np.bool_, copy=False)
        coarse_map = coarse_map.copy()
        coarse_map[~coarse_mask] = np.nan

        summary_features = _native_count_log_dispersion_feature(
            native_map,
            native_mask
        )
        summary_mask = np.ones(summary_features.shape, dtype=np.bool_)

        hybrid = np.concatenate([coarse_map, summary_features])
        hybrid_mask = np.concatenate([coarse_mask, summary_mask])
        return (
            hybrid.astype(np.float32, copy=False),
            hybrid_mask.astype(np.bool_, copy=False)
        )

    def _build_hybrid_batch_from_native(
            native_maps: NDArray,
            native_masks: NDArray[np.bool_],
            downscale_nside: int
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        outputs = [
            _build_hybrid_sample_from_native(
                native_map,
                native_mask,
                downscale_nside=downscale_nside
            )
            for native_map, native_mask in zip(native_maps, native_masks)
        ]
        x = np.stack([output[0] for output in outputs], axis=0)
        mask = np.stack([output[1] for output in outputs], axis=0)
        return x, mask

    def _generate_native_with_callable(
            catwise_model: Catwise,
            sim_callable,
            *args,
            **kwargs
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        original_downscale_nside = catwise_model.downscale_nside
        try:
            catwise_model.downscale_nside = None
            return sim_callable(*args, **kwargs)
        finally:
            catwise_model.downscale_nside = original_downscale_nside

    def _make_real_sample_native(
            catwise_model: Catwise
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        original_downscale_nside = catwise_model.downscale_nside
        try:
            catwise_model.downscale_nside = None
            return catwise_model.make_real_sample()
        finally:
            catwise_model.downscale_nside = original_downscale_nside

    def make_native_count_summary_simulator(
            catwise_model: Catwise,
            sim_callable,
            downscale_nside: int
    ):
        def simulator_with_summary(
                rng_key: Optional[NPKey] = None,
                **kwargs
        ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
            native_map, native_mask = _generate_native_with_callable(
                catwise_model,
                sim_callable,
                rng_key=rng_key,
                **kwargs
            )
            return _build_hybrid_sample_from_native(
                native_map,
                native_mask,
                downscale_nside=downscale_nside
            )

        return simulator_with_summary

    def _batch_generate_dipole_native(
            jax_model: CatwiseJax,
            theta: dict[str, NDArray],
            key: jax.Array,
            batch_size: int
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        original_downscale_nside = jax_model.downscale_nside
        try:
            jax_model.downscale_nside = None
            return jax_model.batch_generate_dipole(
                theta,
                key,
                batch_size=batch_size,
                show_progress=True
            )
        finally:
            jax_model.downscale_nside = original_downscale_nside

    def make_jax_model_sim_wrapper(
            jax_model: CatwiseJax,
            fixed_kwargs: dict[str, Any],
            catwise_version: str,
            batch_size: int,
            append_native_count_summary: bool = False
    ):
        def model_sim_wrapper(
                npkey: NPKey,
                params: dict[str, NDArray],
                noise: bool = True,
                ui: Optional[MultiRoundInfererUI] = None
        ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
            n_sims = _n_sims_from_params(params)
            theta = {
                key: _batch_parameter(value, n_sims)
                for key, value in fixed_kwargs.items()
            }
            theta.update({key: np.asarray(value) for key, value in params.items()})
            theta['w1_max'] = np.full(
                n_sims,
                16.5 if catwise_version == 'S22' else 16.4,
                dtype=np.float64,
            )

            jax_key = _jax_key_from_npkey(npkey)
            if append_native_count_summary:
                if jax_model.downscale_nside is None:
                    raise ValueError(
                        'append_native_count_summary requires model.downscale_nside.'
                    )
                downscale_nside = jax_model.downscale_nside
                native_maps, native_masks = _batch_generate_dipole_native(
                    jax_model,
                    theta,
                    jax_key,
                    batch_size=batch_size
                )
                return _build_hybrid_batch_from_native(
                    native_maps,
                    native_masks,
                    downscale_nside=downscale_nside
                )

            return jax_model.batch_generate_dipole(
                theta,
                jax_key,
                batch_size=batch_size,
                show_progress=True
            )

        return model_sim_wrapper

    prior = DipolePriorNP(
        mean_count_range=[np.log10(30_000_000), np.log10(40_000_000)], #U[7.5,7.6]
        speed_range=[0, 8] if not WIDE_V_PRIOR else [0, 20]
    )
    prior.change_kwarg(
        param_short_name='N',
        new_kwarg='log10_n_initial_samples'
    )

    def add_error_scale(
            prior: DipolePriorNP,
            use_common_error: bool
    ) -> None:
        prior.add_prior(
            short_name='etaWX' if use_common_error else 'etaW1',
            simulator_kwarg='w1_extra_error',
            low=0,
            high=8, # U[0,5]
            dist_type='uniform',
            index=1
        )
        if not use_common_error:
            prior.add_prior(
                short_name='etaW2',
                simulator_kwarg='w2_extra_error',
                low=0,
                high=8, # U[0,5]
                dist_type='uniform',
                index=2
            )

    def add_tdist_shape_param(prior: DipolePriorNP):
        prior.add_prior(
            short_name='nu',
            simulator_kwarg='log10_magnitude_error_shape_param',
            low=0.3,
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

    def add_poisson_cluster_params(prior: DipolePriorNP):
        prior.add_prior(
            short_name='lambda_clus',
            simulator_kwarg='lambda_clus',
            low=0,
            high=8,
            dist_type='uniform',
        )

    def add_confusion_params(prior: DipolePriorNP):
        prior.add_prior(
            short_name='log10_kw1',
            simulator_kwarg='log10_w1conf_scale',
            low=-2,
            high=2,
            dist_type='uniform',
        )
        prior.add_prior(
            short_name='log10_kw2',
            simulator_kwarg='log10_w2conf_scale',
            low=-2,
            high=2,
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

    if SIMULATE_CLUSTERING == 'poisson':
        add_poisson_cluster_params(prior)
        theta_0['lambda_clus'] = 0.

    if not COMMON_ERROR:
        theta_0['w2_extra_error'] = 4.

    if ADD_CONFUSION:
        add_confusion_params(prior)
        theta_0['log10_w1conf_scale'] = 1.
        theta_0['log10_w2conf_scale'] = 1.

    match args.model:
        case 'free_gauss_extra_err':
            ERROR_DIST = 'gaussian'
            add_error_scale(prior, COMMON_ERROR)
            # theta_0.pop('w2_extra_error')

        case 'free_students-t_extra_err':
            ERROR_DIST = 'students-t'
            add_error_scale(prior, COMMON_ERROR)
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
            # the default params for the dipole are the CMB ones
            simulator = simulator_wrapper
            prior.remove_prior('D')
            prior.remove_prior('phi')
            prior.remove_prior('theta')
            add_error_scale(prior, COMMON_ERROR)
            theta_0.pop('observer_speed')
            theta_0.pop('dipole_longitude')
            theta_0.pop('dipole_latitude')
            if COMMON_ERROR:
                assert prior.ndim == 2
            else:
                assert prior.ndim == 3

        case 'cmb_direction':
            ERROR_DIST = 'gaussian'
            simulator = simulator_wrapper
            prior.remove_prior('phi')
            prior.remove_prior('theta')
            add_error_scale(prior, COMMON_ERROR)
            theta_0.pop('dipole_longitude'); theta_0.pop('dipole_latitude')
            if COMMON_ERROR:
                assert prior.ndim == 3
            else:
                assert prior.ndim == 4

        case 'cmb_velocity':
            ERROR_DIST = 'gaussian'
            simulator = simulator_wrapper
            prior.remove_prior('D')
            add_error_scale(prior, COMMON_ERROR)
            theta_0.pop('observer_speed')
            if COMMON_ERROR:
                assert prior.ndim == 4
            else:
                assert prior.ndim == 5

        case 'secrest+21':
            ERROR_DIST = 'gaussian'
            simulator = partial(
                simulator_wrapper,
                observer_speed=2.156, # from CatWISE_Dipole_results.ipynb in secrest's code
                dipole_longitude=238.2,
                dipole_latitude=28.8
            )
            prior.remove_prior('D')
            prior.remove_prior('phi')
            prior.remove_prior('theta')
            add_error_scale(prior, COMMON_ERROR)
            theta_0.pop('observer_speed')
            theta_0.pop('dipole_longitude')
            theta_0.pop('dipole_latitude')
            if COMMON_ERROR:
                assert prior.ndim == 2
            else:
                assert prior.ndim == 3

        case 'dam+23':
            ERROR_DIST = 'gaussian'
            simulator = partial(
                simulator_wrapper,
                observer_speed=2.68, # from CatWISE_Dipole_results.ipynb in secrest's code
                dipole_longitude=237.2,
                dipole_latitude=41.8
            )
            prior.remove_prior('D')
            prior.remove_prior('phi')
            prior.remove_prior('theta')
            add_error_scale(prior, COMMON_ERROR)
            theta_0.pop('observer_speed')
            theta_0.pop('dipole_longitude')
            theta_0.pop('dipole_latitude')
            if COMMON_ERROR:
                assert prior.ndim == 2
            else:
                assert prior.ndim == 3

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
                map_ndim = (
                    12 * current_downscale**2
                    if APPEND_NATIVE_COUNT_SUMMARY and current_downscale is not None
                    else None
                )
                summary_ndim = 1 if APPEND_NATIVE_COUNT_SUMMARY else None
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
                        'n_likelihood_samples':  10_000,
                        'map_ndim': map_ndim,
                        'summary_ndim': summary_ndim,
                        'native_count_summary': (
                            'log_dispersion'
                            if APPEND_NATIVE_COUNT_SUMMARY else None
                        )
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
            chunk_size=args.chunk_size,
            use_common_extra_error=COMMON_ERROR,
            model_identifier=args.model,
            downscale_nside=current_downscale,
            base_mask_version=args.catwise_version,
            generate_correlated_points=USE_CLUSTERS,
            s21_catalogue_path='/home/oliver/Documents/catsim/src/catsim/data/catwise_agns_masked_final_w1lt16p5_alpha.fits',
            use_noecl_mask=NOECL_MASK,
            add_confusion_noise=ADD_CONFUSION,
            max_cluster_children_per_parent=args.max_children
        )

        model = CatwiseJax(config) if args.use_jax else Catwise(config)
        model.initialise_data()

        if args.use_jax:
            model_sim_wrapper = make_jax_model_sim_wrapper(
                model,
                fixed_kwargs=_partial_keywords(simulator),
                catwise_version=args.catwise_version,
                batch_size=args.jax_batch_size,
                append_native_count_summary=APPEND_NATIVE_COUNT_SUMMARY
            )
        else:
            sim_callable = (
                make_native_count_summary_simulator(
                    model,
                    simulator,
                    downscale_nside=current_downscale
                )
                if APPEND_NATIVE_COUNT_SUMMARY else simulator
            )

            def model_sim_wrapper(
                    npkey: NPKey,
                    params: dict[str, NDArray],
                    noise: bool = True,
                    ui: Optional[MultiRoundInfererUI] = None
            ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
                return batch_simulate(
                    params,
                    sim_callable,
                    n_workers=N_WORKERS,
                    ui=ui,
                    rng_key=npkey
                )

        data_model = Catwise(config) if args.use_jax else model
        if args.use_jax:
            data_model.initialise_data()

        if args.catwise_version == 'S21':
            if APPEND_NATIVE_COUNT_SUMMARY:
                native_x0, native_mask = _make_real_sample_native(data_model)
                x0, mask = _build_hybrid_sample_from_native(
                    native_x0,
                    native_mask,
                    downscale_nside=current_downscale
                )
            else:
                x0, mask = data_model.make_real_sample()
        elif args.catwise_version == 'S22':
            x0 = np.asarray(
                np.load('dipolesbi/catwise/catwise_S22.npy'), dtype=np.float32
            )
            mask = data_model.binary_mask
            x0[~mask] = np.nan

            if APPEND_NATIVE_COUNT_SUMMARY:
                x0, mask = _build_hybrid_sample_from_native(
                    x0,
                    mask,
                    downscale_nside=current_downscale
                )
            elif current_downscale:
                x0, mask = downgrade_ignore_nan(x0, mask, current_downscale)
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
