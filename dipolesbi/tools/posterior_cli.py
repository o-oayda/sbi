from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from .posterior_samples import PosteriorSamplesInterface, save_corner_plot


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect posterior samples produced by the multi-round inferer.",
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Directory containing samples_rnd-*.csv files (e.g. fiducial_50k/<run>).",
    )
    parser.add_argument(
        "--round",
        type=int,
        dest="round_id",
        help="Round index to load. If omitted you will be prompted to choose.",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Skip the interactive table that lists available rounds.",
    )
    parser.add_argument(
        "--limit-columns",
        type=int,
        default=None,
        help="Only show statistics for the first N columns.",
    )
    parser.add_argument(
        "--corner",
        type=Path,
        help="If set, save a GetDist corner plot to this path.",
    )
    parser.add_argument(
        "--corner-columns",
        nargs="+",
        default=None,
        help="Subset of columns to include in the corner plot.",
    )
    parser.add_argument(
        "--corner-markers",
        type=float,
        nargs="+",
        default=None,
        help="Optional marker values (e.g. ground truth) for the corner plot.",
    )
    parser.add_argument(
        "--corner-unfilled",
        action="store_true",
        help="Render corner contours without filling.",
    )
    parser.add_argument(
        "--ppc-count",
        type=int,
        default=None,
        help="Number of posterior predictive simulations to generate.",
    )
    parser.add_argument(
        "--ppc-output",
        type=Path,
        default=None,
        help="Save posterior predictive plot to this path (defaults to run directory).",
    )
    parser.add_argument(
        "--ppc-seed",
        type=int,
        default=None,
        help="Seed for posterior predictive sampling and simulation.",
    )
    parser.add_argument(
        "--ppc-workers",
        type=int,
        default=None,
        help="Number of worker processes for posterior predictive simulations (defaults to available CPUs).",
    )
    parser.add_argument(
        "--ppc-downscale",
        type=int,
        default=None,
        help="Optional downscaled nside for posterior predictive simulations (default native 64).",
    )
    parser.add_argument(
        "--sky-prob",
        type=Path,
        default=None,
        help="If set, save a sky probability (dipole direction) plot to this path.",
    )
    parser.add_argument(
        "--sky-nside",
        type=int,
        default=64,
        help="Healpix nside used to bin samples for the sky probability map.",
    )
    parser.add_argument(
        "--sky-smooth",
        type=float,
        default=0.05,
        help="Gaussian smoothing sigma (radians) applied to the sky probability map.",
    )
    parser.add_argument(
        "--sky-lon-col",
        default="dipole_longitude",
        help="Column name containing dipole longitudes in degrees.",
    )
    parser.add_argument(
        "--sky-lat-col",
        default="dipole_latitude",
        help="Column name containing dipole latitudes in degrees.",
    )
    parser.add_argument(
        "--sky-truth",
        type=float,
        nargs=2,
        metavar=("LON", "LAT"),
        default=None,
        help="Optional true dipole longitude/latitude in degrees for annotation.",
    )
    return parser


def _format_number(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.4g}"


def _column_summary_table(samples, columns: list[str] | None = None) -> Table:
    table = Table(title="Column summary", expand=True)
    table.add_column("Column")
    table.add_column("Finite", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    column_names = columns if columns is not None else list(samples.columns)

    for name in column_names:
        if name not in samples.data:
            continue
        values = np.asarray(samples[name], dtype=np.float64)
        finite_mask = np.isfinite(values)
        finite_count = int(np.count_nonzero(finite_mask))
        if finite_count:
            finite_values = values[finite_mask]
            mean = float(np.mean(finite_values))
            std = float(np.std(finite_values))
            vmin = float(np.min(finite_values))
            vmax = float(np.max(finite_values))
        else:
            mean = std = vmin = vmax = float("nan")
        table.add_row(
            name,
            str(finite_count),
            _format_number(mean),
            _format_number(std),
            _format_number(vmin),
            _format_number(vmax),
        )
    return table


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    console = Console()
    interface = PosteriorSamplesInterface(args.experiment_dir, console=console)
    samples = interface.load_round(
        round_id=args.round_id,
        show_table=not args.no_table,
    )

    console.print(
        f"[green]Loaded round {samples.round_id} "
        f"from {samples.info.path} with {samples.n_samples} samples.[/green]"
    )

    columns = list(samples.columns)
    if args.limit_columns is not None:
        columns = columns[: args.limit_columns]

    console.print(_column_summary_table(samples, columns=columns))

    logz_mean, logz_err = samples.log_evidence()
    if logz_err is not None:
        console.print(f"logZ = {logz_mean:.4f} ± {logz_err:.4f}")
    else:
        console.print(f"logZ = {logz_mean:.4f} (uncertainty unavailable)")

    if args.corner is not None:
        filled = not args.corner_unfilled
        corner_path = save_corner_plot(
            samples,
            args.corner,
            param_columns=args.corner_columns,
            markers=args.corner_markers,
            filled=filled,
        )
        console.print(f"[blue]Corner plot written to {corner_path}[/blue]")

    if args.ppc_count is not None:
        if args.ppc_count <= 0:
            console.print("[red]--ppc-count must be positive.[/red]")
        else:
            try:
                result = interface.posterior_predictive(
                    count=args.ppc_count,
                    round_id=samples.round_id,
                    seed=args.ppc_seed,
                    workers=args.ppc_workers,
                    output_path=args.ppc_output,
                    show_table=False,
                    downscale_nside=args.ppc_downscale,
                )
            except NotImplementedError as exc:
                console.print(f"[red]{exc}[/red]")
            else:
                if len(result.output_paths) == 1:
                    console.print(
                        f"[green]Posterior predictive plot written to {result.output_paths[0]}[/green]"
                    )
                else:
                    console.print(
                        f"[green]Posterior predictive plots written to {len(result.output_paths)} files:[/green]"
                    )
                    for path in result.output_paths:
                        console.print(f"  - {path}")
    if args.sky_prob is not None:
        try:
            truth_tuple = tuple(args.sky_truth) if args.sky_truth is not None else None
            sky_path = interface.plot_sky_probability(
                args.sky_prob,
                round_id=samples.round_id,
                lon_column=args.sky_lon_col,
                lat_column=args.sky_lat_col,
                nside=args.sky_nside,
                smooth=args.sky_smooth,
                truth_deg=truth_tuple,
                show_table=False,
            )
        except Exception as exc:
            console.print(f"[red]Failed to generate sky probability plot: {exc}[/red]")
        else:
            console.print(f"[green]Sky probability plot written to {sky_path}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
