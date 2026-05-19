from typing import TYPE_CHECKING, Any

__all__ = [
    "PosteriorRepository",
    "PosteriorRunInfo",
    "PosteriorSamples",
    "PosteriorSamplesInterface",
    "PosteriorPredictiveResult",
    "format_posterior_samples",
    "load_model_config",
    "sample_posterior_csv",
    "save_corner_plot",
    "save_model_config",
]

_EXPORT_MODULES = {
    "load_model_config": ".model_config_io",
    "save_model_config": ".model_config_io",
}

if TYPE_CHECKING:  # pragma: no cover
    from .model_config_io import load_model_config, save_model_config
    from .posterior_samples import (
        PosteriorPredictiveResult,
        PosteriorRepository,
        PosteriorRunInfo,
        PosteriorSamples,
        PosteriorSamplesInterface,
        format_posterior_samples,
        sample_posterior_csv,
        save_corner_plot,
    )


def __getattr__(name: str) -> Any:
    if name in __all__:
        from importlib import import_module

        module = import_module(_EXPORT_MODULES.get(name, ".posterior_samples"), __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
