from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class QuantileSummaryDiagnosticSpec:
    """Location and plotting metadata for one flattened quantile summary."""

    name: str
    start: int
    bin_edges: tuple[float, ...]
    quantiles: tuple[float, ...]
    x_label: str

    @property
    def n_bins(self) -> int:
        return len(self.bin_edges) - 1

    @property
    def width(self) -> int:
        return self.n_bins * len(self.quantiles)

    @property
    def stop(self) -> int:
        return self.start + self.width

    @property
    def bin_centres(self) -> NDArray[np.float64]:
        edges = np.asarray(self.bin_edges, dtype=np.float64)
        return 0.5 * (edges[:-1] + edges[1:])


def plot_round_quantile_diagnostics(
    round_idx: int,
    data: tuple[NDArray[np.float32], NDArray[np.bool_]],
    reference_data: NDArray,
    reference_mask: NDArray[np.bool_],
    output_dir: str,
    *,
    specs: Sequence[QuantileSummaryDiagnosticSpec],
) -> list[Path]:
    """Plot reference quantiles against current-round simulation moments."""
    simulations, simulation_mask = (np.asarray(value) for value in data)
    reference = np.asarray(reference_data)
    reference_valid = np.asarray(reference_mask, dtype=bool)

    if simulations.ndim != 2 or simulation_mask.shape != simulations.shape:
        raise ValueError("Round diagnostic simulations and masks must be matching 2D arrays.")
    if reference.ndim != 1 or reference_valid.shape != reference.shape:
        raise ValueError("Round diagnostic reference data and mask must be matching 1D arrays.")
    if simulations.shape[1] != reference.size:
        raise ValueError("Round diagnostic simulations must match the reference dimension.")

    plot_dir = Path(output_dir) / "round_summary_diagnostics"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for spec in specs:
        if spec.start < 0 or spec.stop > reference.size:
            raise ValueError(f"Summary diagnostic {spec.name!r} lies outside the data vector.")
        if spec.n_bins < 1 or not spec.quantiles:
            raise ValueError(f"Summary diagnostic {spec.name!r} has an invalid shape.")

        summary_slice = slice(spec.start, spec.stop)
        summary_shape = (spec.n_bins, len(spec.quantiles))
        reference_summary = reference[summary_slice].reshape(summary_shape)
        reference_summary = np.where(
            reference_valid[summary_slice].reshape(summary_shape),
            reference_summary,
            np.nan,
        )
        simulation_summary = simulations[:, summary_slice].reshape(
            simulations.shape[0],
            *summary_shape,
        )
        simulation_summary = np.where(
            simulation_mask[:, summary_slice].reshape(
                simulations.shape[0],
                *summary_shape,
            ),
            simulation_summary,
            np.nan,
        )
        simulation_mean = np.nanmean(simulation_summary, axis=0)
        simulation_std = np.nanstd(
            simulation_summary,
            axis=0,
            ddof=1 if simulations.shape[0] > 1 else 0,
        )

        n_quantiles = len(spec.quantiles)
        n_columns = min(2, n_quantiles)
        n_rows = (n_quantiles + n_columns - 1) // n_columns
        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(6.5 * n_columns, 3.2 * n_rows),
            sharex=True,
            squeeze=False,
        )
        flat_axes = axes.ravel()
        for quantile_idx, quantile in enumerate(spec.quantiles):
            axis = flat_axes[quantile_idx]
            mean = simulation_mean[:, quantile_idx]
            std = simulation_std[:, quantile_idx]
            axis.plot(
                spec.bin_centres,
                reference_summary[:, quantile_idx],
                marker="o",
                color="tab:orange",
                label="Reference x0",
            )
            axis.plot(
                spec.bin_centres,
                mean,
                marker="o",
                color="tab:green",
                label="Round simulations mean",
            )
            axis.fill_between(
                spec.bin_centres,
                mean - std,
                mean + std,
                color="tab:green",
                alpha=0.2,
                label="Round simulations ±1 std",
            )
            axis.set_title(f"Quantile {quantile:g}")
            axis.set_xlabel(spec.x_label)
            axis.set_ylabel("Flux [mJy]")
            axis.legend()

        for axis in flat_axes[n_quantiles:]:
            axis.set_visible(False)

        fig.suptitle(f"{spec.name.replace('_', ' ')}, inference round {round_idx + 1}")
        fig.tight_layout()
        output_path = plot_dir / f"round_{round_idx:02d}_{spec.name}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def make_round_quantile_diagnostic(
    specs: Sequence[QuantileSummaryDiagnosticSpec],
) -> Callable[..., None]:
    """Create the optional callback consumed by ``MultiRoundInferer``."""
    frozen_specs = tuple(specs)

    def diagnostic(*args, **kwargs) -> None:
        plot_round_quantile_diagnostics(*args, specs=frozen_specs, **kwargs)

    return diagnostic
