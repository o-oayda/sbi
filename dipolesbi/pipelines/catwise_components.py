"""Components shared by CatWISE observation preparation and inference."""

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
from catsim import CatwiseConfig
from catsim.utils.healsphere import downgrade_ignore_nan

from dipolesbi.tools.configs import (
    DataTransformSpec,
    Scenario,
    ThetaTransformSpec,
)
from dipolesbi.tools.np_rngkey import NPKey
from dipolesbi.tools.priors_np import DipolePriorNP


def build_prior(run: Mapping[str, Any]) -> DipolePriorNP:
    """Construct the ordered prior exactly as recorded in the run contract."""
    prior = DipolePriorNP()
    for name in list(prior.prior_names):
        prior.remove_prior(name)
    for entry in run['priors']:
        prior.add_prior(**dict(entry))
    return prior


def build_catwise_config(run: Mapping[str, Any]) -> CatwiseConfig:
    """Configure a native-resolution simulator; shared processing follows it."""
    model = dict(run['model'])
    clustering = model.pop('simulate_clustering')
    if clustering != 'poisson':
        raise ValueError('Only Poisson clustering is supported.')
    paths, obs, execution = run['paths'], run['observation'], run['execution']
    return CatwiseConfig(
        **model,
        chunk_size=execution['chunk_size'],
        downscale_nside=None,
        base_mask_version=obs['base_mask_version'],
        s21_catalogue_path=paths['s21_catalogue'],
        source_catalogue_path=paths.get('source_catalogue'),
        hashgrid_cache_dir=paths.get('hashgrid_cache'),
    )


def build_scenario(
    run: Mapping[str, Any], prior: DipolePriorNP, out_dir: str
) -> Scenario:
    inference, execution = run['inference'], run['execution']
    if inference['mode'] != 'NLE' or inference['scenario'] != 'anynside_nle':
        raise ValueError('Only the anynside NLE scenario is supported.')
    mr = dict(inference['multiround'])
    for key in (
        'simulations_per_round',
        'effective_simulations',
        'simulation_path',
    ):
        mr.pop(key, None)
    mr.update(
        prng_integer_seed=execution['seed'],
        plot_save_dir=str(out_dir),
        simulation_budget=execution['requested_simulations'],
        n_rounds=execution['n_rounds'],
        reference_theta=run['reference_theta'],
    )
    data = dict(inference['data_transform'])
    if data.get('embedding_config') is not None:
        from dipolesbi.tools.configs import EmbeddingNetConfig

        data['embedding_config'] = EmbeddingNetConfig(
            **data['embedding_config']
        )
    return Scenario.anynside_nle(
        nside=run['observation']['downscale_nside'],
        theta_prior=prior.to_jax(),
        reference_theta=run['reference_theta'],
        training_overrides=dict(inference['training']),
        multiround_overrides=mr,
        flow_overrides=dict(inference['flow']),
        data_spec=DataTransformSpec(**data),
        theta_spec=ThetaTransformSpec(**inference['theta_transform']),
    )


def process_native_sample(native_map, native_mask, observation):
    """Downgrade a native count map and append its log dispersion."""
    native_map = np.asarray(native_map)
    native_mask = np.asarray(native_mask, dtype=bool)
    expected = 12 * observation['native_nside'] ** 2
    if native_map.shape != (expected,) or native_mask.shape != (expected,):
        raise ValueError(
            f'Native map and mask must both have shape ({expected},).'
        )
    if observation['native_count_summary'] != 'log_dispersion':
        raise ValueError('Only log_dispersion native summary is supported.')
    valid = native_mask & np.isfinite(native_map)
    counts = np.rint(native_map[valid]).astype(np.float64)
    if counts.size < 2 or np.any(counts < 0):
        raise ValueError(
            'Native counts must contain at least two nonnegative pixels.'
        )
    mean = float(np.mean(counts))
    if mean <= 0:
        raise ValueError('Native count mean must be positive.')
    dispersion = float(np.var(counts, ddof=1) / mean)
    if dispersion <= 0:
        raise ValueError('Native count dispersion must be positive.')
    coarse, mask = downgrade_ignore_nan(
        native_map, native_mask, observation['downscale_nside']
    )
    coarse = np.asarray(coarse, dtype=np.float32).copy()
    mask = np.asarray(mask, dtype=bool)
    coarse[~mask] = np.nan
    return (
        np.concatenate(
            (coarse, np.array([np.log(dispersion)], dtype=np.float32))
        ),
        np.concatenate((mask, np.ones(1, dtype=bool))),
    )


def make_model_sim_wrapper(
    model, run: Mapping[str, Any], expected_native_mask=None
):
    """Adapt either catsim backend to the inferer's batched simulator interface."""
    obs, execution = run['observation'], run['execution']
    fixed = dict(run['fixed_parameters'])
    cuts = {name: obs[name] for name in ('w1_min', 'w1_max', 'w12_min')}
    if fixed.keys() & cuts.keys():
        raise ValueError('Fixed parameters conflict with observation cuts.')
    fixed.update(cuts)
    if getattr(model, 'downscale_nside', None) is not None:
        raise ValueError('Simulator must produce native-resolution maps.')
    expected_mask = (
        None
        if expected_native_mask is None
        else np.asarray(expected_native_mask, dtype=bool)
    )

    def process(x, mask):
        if expected_mask is not None and not np.array_equal(
            mask, expected_mask
        ):
            raise ValueError(
                'Simulator native mask does not match the observed mask.'
            )
        return process_native_sample(x, mask, obs)

    def wrapper(npkey, params, noise=True, ui=None):
        overlap = fixed.keys() & params.keys()
        if overlap:
            raise ValueError(
                f'Fixed simulator parameters conflict with inferred parameters: {sorted(overlap)}'
            )
        arrays = {k: np.asarray(v) for k, v in params.items()}
        lengths = {v.shape[0] for v in arrays.values() if v.ndim == 1}
        if any(v.ndim > 1 for v in arrays.values()) or len(lengths) > 1:
            raise ValueError(
                'Simulator parameters must be scalars or equally sized vectors.'
            )
        n = next(iter(lengths), 1)
        if n < 1:
            raise ValueError('Simulator batch must not be empty.')

        def batch(value):
            if value is None:
                value = np.nan
            value = np.asarray(value)
            if value.ndim == 0:
                return np.full(n, value.item())
            if value.shape != (n,):
                raise ValueError(
                    f'Simulator parameter must be scalar or shape ({n},).'
                )
            return value

        theta = {k: batch(v) for k, v in {**fixed, **arrays}.items()}
        if execution['backend'] == 'jax':
            key = (
                npkey._ss.generate_state(2, dtype=np.uint32)
                if isinstance(npkey, NPKey)
                else np.asarray(npkey, dtype=np.uint32).reshape(2)
            )
            maps, masks = model.batch_generate_dipole(
                theta,
                jax.device_put(key),
                batch_size=execution['jax_batch_size'],
                show_progress=ui is None,
            )
            if len(maps) != n or len(masks) != n:
                raise ValueError(
                    'Simulator returned an unexpected batch size.'
                )
            results = [process(x, mask) for x, mask in zip(maps, masks)]
        elif execution['backend'] == 'numpy':
            if not isinstance(npkey, NPKey):
                npkey = NPKey.from_jax(npkey)
            results = [
                process(
                    *model.generate_dipole(
                        **{
                            k: (
                                None
                                if k in fixed and fixed[k] is None
                                else v[i]
                            )
                            for k, v in theta.items()
                        },
                        rng_key=npkey.fold_in(i),
                    )
                )
                for i in range(n)
            ]
        else:
            raise ValueError(
                f"Unknown simulator backend: {execution['backend']}"
            )
        return np.stack([x for x, _ in results]), np.stack(
            [m for _, m in results]
        )

    return wrapper
