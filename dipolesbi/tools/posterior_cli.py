from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
from getdist import plots

from .posterior_samples import PosteriorSamplesInterface, PosteriorSamples


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect posterior samples produced by the multi-round inferer.",
    )
    parser.add_argument(
        "experiment_dirs",
        type=Path,
        nargs="+",
        help="One or more directories containing samples_rnd-*.csv files.",
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
    parser.add_argument(
        "--legend",
        nargs="+",
        help="Legend labels corresponding to each experiment directory.",
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
    interfaces = [PosteriorSamplesInterface(path, console=console) for path in args.experiment_dirs]

    samples_list: list[PosteriorSamples] = []
    if args.legend is not None:
        if len(args.legend) != len(args.experiment_dirs):
            raise ValueError("Number of legend labels must match number of experiment directories.")
        labels = list(args.legend)
    else:
        labels = [path.name for path in args.experiment_dirs]

    for idx, iface in enumerate(interfaces):
        samples_i = iface.load_round(
            round_id=args.round_id,
            show_table=not args.no_table,
        )
        samples_list.append(samples_i)
        label = labels[idx]

        console.print(
            f"[green]Loaded round {samples_i.round_id} from {samples_i.info.path} "
            f"with {samples_i.n_samples} samples.[/green]"
        )

        columns_i = list(samples_i.columns)
        if args.limit_columns is not None:
            columns_i = columns_i[: args.limit_columns]

        console.print(_column_summary_table(samples_i, columns=columns_i))

        logz_mean, logz_err = samples_i.log_evidence()
        if logz_err is not None:
            console.print(f"logZ ({label}) = {logz_mean:.4f} ± {logz_err:.4f}")
        else:
            console.print(f"logZ ({label}) = {logz_mean:.4f} (uncertainty unavailable)")

    if not samples_list:
        console.print("[red]No samples loaded.[/red]")
        return 1

    primary_samples = samples_list[0]
    primary_interface = interfaces[0]
    summary_columns = list(primary_samples.columns)
    if args.limit_columns is not None:
        summary_columns = summary_columns[: args.limit_columns]

    if args.corner is not None:
        filled = not args.corner_unfilled

        if args.corner_columns is not None:
            requested_params = list(args.corner_columns)
            missing = [col for col in requested_params if any(col not in s.data for s in samples_list)]
            if missing:
                raise ValueError(f"Corner columns {missing} not found in all runs.")
            param_columns = requested_params
        else:
            base_params = primary_samples.parameter_columns()
            common_params = [col for col in base_params if all(col in s.data for s in samples_list)]
            if not common_params:
                raise ValueError("No common parameter columns found across runs for corner plot.")
            param_columns = common_params

        mc_samples_list = []
        for s, label in zip(samples_list, labels):
            mc = s.to_getdist(param_columns=param_columns, weight_column="weights")
            mc.label = label  # type: ignore[attr-defined]
            mc_samples_list.append(mc)

        plotter = plots.get_subplot_plotter()
        plotter.settings.progress = False
        plotter.triangle_plot(
            mc_samples_list,
            params=param_columns,
            filled=filled,
            legend_labels=labels,
            markers=args.corner_markers,
        )
        corner_path = Path(args.corner).expanduser()
        corner_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.export(str(corner_path))
        plt.close("all")
        console.print(f"[blue]Corner plot written to {corner_path}[/blue]")

    if args.ppc_count is not None:
        if args.ppc_count <= 0:
            console.print("[red]--ppc-count must be positive.[/red]")
        else:
            try:
                result = primary_interface.posterior_predictive(
                    count=args.ppc_count,
                    round_id=primary_samples.round_id,
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
            base_cycle = ['cornflowerblue', 'tomato']
            default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
            remaining = [c for c in default_cycle if c not in base_cycle]
            color_cycle = base_cycle + remaining + ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]

            plt.figure(figsize=(8, 4))
            legend_handles = []
            from matplotlib.lines import Line2D

            for idx, (iface, s, label) in enumerate(zip(interfaces, samples_list, labels)):
                color = color_cycle[idx % len(color_cycle)]
                disable_mesh = idx > 0
                no_axes = idx > 0
                truth = truth_tuple if idx == 0 else None
                iface.plot_sky_probability(
                    output_path=None,
                    round_id=s.round_id,
                    lon_column=args.sky_lon_col,
                    lat_column=args.sky_lat_col,
                    nside=args.sky_nside,
                    smooth=args.sky_smooth,
                    truth_deg=truth,
                    show_table=False,
                    disable_mesh=disable_mesh,
                    no_axes=no_axes,
                    show=False,
                    color=color,
                )
                legend_handles.append(Line2D([0], [0], color=color, lw=2, label=label))

            if legend_handles:
                plt.legend(handles=legend_handles, loc="lower left")

            sky_path = Path(args.sky_prob).expanduser()
            sky_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(sky_path, dpi=300, bbox_inches="tight")
            plt.close(plt.gcf())
            console.print(f"[green]Sky probability plot written to {sky_path}[/green]")
        except Exception as exc:
            plt.close(plt.gcf())
            console.print(f"[red]Failed to generate sky probability plot: {exc}[/red]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
