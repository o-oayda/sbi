from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from dipolesbi.catwise.maps import Catwise
from dipolesbi.tools.configs import CatwiseConfig
from dipolesbi.tools.np_rngkey import NPKey

# cache Catwise instances per worker so we only pay setup once
_WORKER_MODELS: dict[Tuple[Tuple[str, Any], ...], Catwise] = {}


def _config_key(config_items: Iterable[Tuple[str, Any]]) -> Tuple[Tuple[str, Any], ...]:
    normalised: list[Tuple[str, Any]] = []
    for key, value in config_items:
        if isinstance(value, list):
            value = tuple(value)
        normalised.append((key, value))
    return tuple(sorted(normalised))


def _get_model(config_dict: Dict[str, Any]) -> Catwise:
    key = _config_key(config_dict.items())
    model = _WORKER_MODELS.get(key)
    if model is None:
        cfg = CatwiseConfig(**config_dict)
        model = Catwise(cfg)
        model.initialise_data()
        _WORKER_MODELS[key] = model
    return model


def remote_generate_dipole(
    config_dict: Dict[str, Any],
    simulator_kwargs: Dict[str, Any],
    rng_key: NPKey | None,
) -> Tuple[Any, Any]:
    model = _get_model(config_dict)
    try:
        return model.generate_dipole(rng_key=rng_key, **simulator_kwargs)
    except ValueError as exc:
        debug = {
            'n_samples': getattr(model, 'n_samples', None),
            'chunk_size': getattr(model, 'chunk_size', None),
            'log10_n_initial_samples': simulator_kwargs.get('log10_n_initial_samples'),
            'model_config': config_dict,
        }
        raise ValueError(f'remote_generate_dipole failed: {debug}') from exc
