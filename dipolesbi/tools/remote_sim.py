from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Iterable, Tuple

from dipolesbi.catwise.maps import Catwise
from dipolesbi.tools.configs import CatwiseConfig
from dipolesbi.tools.np_rngkey import NPKey

# Cache one Catwise per worker process; each entry carries a lock so threads
# within the same process do not clobber the shared instance.
_WORKER_MODELS: dict[Tuple[Tuple[str, Any], ...], tuple[Catwise, Lock]] = {}


def _config_key(config_items: Iterable[Tuple[str, Any]]) -> Tuple[Tuple[str, Any], ...]:
    normalised: list[Tuple[str, Any]] = []
    for key, value in config_items:
        if isinstance(value, list):
            value = tuple(value)
        normalised.append((key, value))
    return tuple(sorted(normalised))


def _get_model(config_dict: Dict[str, Any]) -> tuple[Catwise, Lock]:
    key = _config_key(config_dict.items())
    entry = _WORKER_MODELS.get(key)
    if entry is None:
        cfg = CatwiseConfig(**config_dict)
        model = Catwise(cfg)
        model.initialise_data()
        entry = (model, Lock())
        _WORKER_MODELS[key] = entry
    return entry[0], entry[1]


def remote_generate_dipole(
    config_dict: Dict[str, Any],
    simulator_kwargs: Dict[str, Any],
    rng_key: NPKey | None,
) -> Tuple[Any, Any]:
    model, lock = _get_model(config_dict)
    # Dask may hand multiple tasks to the same worker process. Guard the cached
    # Catwise so concurrent threads on that worker do not race on its mutable
    # attributes; parallelism still comes from multiple worker processes.
    with lock:
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
