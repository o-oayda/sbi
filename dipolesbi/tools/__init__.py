from typing import TYPE_CHECKING, Any

__all__ = [
    "PosteriorRepository",
    "PosteriorRunInfo",
    "PosteriorSamples",
    "PosteriorSamplesInterface",
    "PosteriorPredictiveResult",
    "save_corner_plot",
]

if TYPE_CHECKING:  # pragma: no cover
    from .posterior_samples import (
        PosteriorPredictiveResult,
        PosteriorRepository,
        PosteriorRunInfo,
        PosteriorSamples,
        PosteriorSamplesInterface,
        save_corner_plot,
    )


def __getattr__(name: str) -> Any:
    if name in __all__:
        from importlib import import_module

        module = import_module(".posterior_samples", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
