from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from getdist import plots
from anesthetic import read_csv as nested_read_csv
from dipolesbi.style import paperplot

from .posterior_samples import (
    PosteriorSamplesInterface,
    PosteriorSamples,
    PosteriorRunInfo,
)
from .utils import sigma_to_prob1D
from .plotting import (
    marker_cycle,
    SKY_PROBABILITY_COLOR_CYCLE,
    get_top_quadrant_bbox,
    quad_tick_labels,
)

plots.set_active_style(paperplot.style_name)


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
        "--corner-include-true",
        action="store_true",
        help="Include true_samples.csv from each experiment in the corner plot when available.",
    )
    parser.add_argument(
        "--corner-simple-titles",
        action="store_true",
        help="Show only parameter labels on corner plot diagonal titles (hide credible intervals).",
    )
    parser.add_argument(
        "--corner-true-legend",
        nargs="+",
        default=None,
        help="Legend labels for true_samples entries (must match number of experiment directories).",
    )
    parser.add_argument(
        "--corner-sigmas",
        type=float,
        nargs="+",
        default=None,
        help="2D Gaussian sigma levels (e.g. 1 2 3) for corner plot contours.",
    )
    parser.add_argument(
        "--corner-unfilled",
        action="store_true",
        help="Render corner contours without filling.",
    )
    parser.add_argument(
        "--corner-no-legend",
        action="store_true",
        help="Suppress the legend in the corner plot.",
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
        default=512,
        help="Healpix nside used to bin samples for the sky probability map.",
    )
    parser.add_argument(
        "--sky-smooth",
        type=float,
        default=0.05,
        help="Gaussian smoothing sigma (radians) applied to the sky probability map.",
    )
    parser.add_argument(
        "--sky-contours",
        type=float,
        nargs="+",
        default=None,
        help="Sigma levels for sky probability contours (e.g. 1 2 3).",
    )
    parser.add_argument(
        "--sky-top-quad",
        action="store_true",
        help="Render the sky probability plot in the top-right Mollweide quadrant (curved boundary).",
    )
    parser.add_argument(
        "--sky-top-quad-legacy",
        action="store_true",
        help="Use the legacy rectangular crop of the top-right quadrant (matches previous behaviour).",
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
        nargs="*",
        default=None,
        help="Optional true dipole directions in degrees as pairs LON LAT (multiple pairs allowed).",
    )
    parser.add_argument(
        "--sky-truth-labels",
        nargs="*",
        default=None,
        help="Legend labels for each sky truth marker (must match number of pairs).",
    )
    parser.add_argument(
        "--legend",
        nargs="+",
        help="Legend labels corresponding to each experiment directory.",
    )
    parser.add_argument(
        "--logz-average-start",
        type=int,
        default=None,
        help="Round index from which to compute weighted logZ averages (applies per run).",
    )
    parser.add_argument(
        "--logz-average-simple",
        action="store_true",
        help="Use simple mean/variance for logZ averaging instead of bootstrap resampling.",
    )
    return parser


def _format_number(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.4g}"


def _load_true_samples(path: Path) -> PosteriorSamples:
    nested = nested_read_csv(path)
    columns = ["sample_index", "weights", *nested.columns]
    data: dict[str, np.ndarray] = {}
    data["sample_index"] = np.arange(len(nested), dtype=np.float64)
    data["weights"] = np.asarray(nested.get_weights(), dtype=np.float64)
    for column in nested.columns:
        data[column] = np.asarray(nested[column], dtype=np.float64)
    info = PosteriorRunInfo(
        round_id=-1,
        path=path,
        n_samples=len(nested),
        columns=tuple(columns),
    )
    return PosteriorSamples(info=info, data=data)


def _simplify_corner_titles(plotter, reference_samples, param_columns: list[str]) -> None:
    subplots = getattr(plotter, "subplots", None)
    if subplots is None:
        return
    try:
        diag_axes = [subplots[i, i] for i in range(len(param_columns))]
    except Exception:
        return

    labels: list[str] = []
    for param in param_columns:
        label = param
        try:
            param_obj = reference_samples.paramNames.parWithName(param)
        except Exception:
            param_obj = None
        if param_obj is not None and getattr(param_obj, "label", None):
            label_value = str(param_obj.label)
            if "$" not in label_value:
                label = f"${label_value}$"
            else:
                label = label_value
        labels.append(label)

    for ax, label in zip(diag_axes, labels):
        if ax is not None:
            ax.set_title(label)


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
    if args.sky_top_quad and args.sky_top_quad_legacy:
        raise ValueError("Cannot combine --sky-top-quad with --sky-top-quad-legacy.")

    console = Console()
    interfaces = [PosteriorSamplesInterface(path, console=console) for path in args.experiment_dirs]

    samples_list: list[PosteriorSamples] = []
    if args.legend is not None:
        if len(args.legend) != len(args.experiment_dirs):
            raise ValueError("Number of legend labels must match number of experiment directories.")
        labels = list(args.legend)
    else:
        labels = [path.name for path in args.experiment_dirs]

    if args.corner_true_legend is not None:
        if len(args.corner_true_legend) != len(args.experiment_dirs):
            raise ValueError(
                "Number of --corner-true-legend entries must match number of experiment directories."
            )
        true_corner_labels = list(args.corner_true_legend)
    else:
        true_corner_labels = None

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

        if args.logz_average_start is not None:
            start_round = args.logz_average_start
            available_rounds = sorted(iface._repository.available_rounds())
            selected_rounds = [r for r in available_rounds if r >= start_round]
            if not selected_rounds:
                console.print(f"[yellow]No rounds ≥ {start_round} found for {label}; skipping logZ average.[/yellow]")
            else:
                table = Table(title=f"logZ per round (≥ {start_round}) for {label}")
                table.add_column("Round", justify="right")
                table.add_column("logZ", justify="right")
                table.add_column("σ", justify="right")

                round_values: list[tuple[float, float | None]] = []
                for r in selected_rounds:
                    round_samples = iface.load_round(round_id=r, show_table=False)
                    z_mean, z_err = round_samples.log_evidence()
                    round_values.append((z_mean, z_err))
                    err_display = f"{z_err:.4f}" if z_err is not None else "N/A"
                    table.add_row(str(r), f"{z_mean:.4f}", err_display)

                console.print(table)

                if not round_values:
                    console.print("[yellow]No valid rounds for logZ averaging.[/yellow]")
                elif len(round_values) == 1:
                    z_mean, z_err = round_values[0]
                    err_display = z_err if (z_err is not None and np.isfinite(z_err)) else 0.0
                    console.print(
                        f"[cyan]Average logZ ({label}): {z_mean:.4f} ± {err_display:.4f}[/cyan]"
                    )
                else:
                    values = np.array([z for z, _ in round_values]) # just z mean from one round
                    if args.logz_average_simple:
                        avg = float(np.mean(values))
                        std = float(np.std(values, ddof=0))
                        console.print(f"[cyan]Average logZ ({label}): {avg:.4f} ± {std:.4f}[/cyan]")
                    else:
                        rng = np.random.default_rng(12345)
                        boot_means = []
                        for _ in range(2000):
                            indices = rng.choice(len(values), size=len(values), replace=True)
                            boot_means.append(values[indices].mean())
                        boot_means = np.asarray(boot_means)
                        avg = float(boot_means.mean())
                        std = float(boot_means.std(ddof=1))
                        console.print(
                            f"[cyan]Bootstrap average logZ ({label}): {avg:.4f} ± {std:.4f}[/cyan]"
                        )

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

        mc_samples_list: list = []
        corner_labels: list[str] = []
        for s, label in zip(samples_list, labels):
            mc = s.to_getdist(param_columns=param_columns, weight_column="weights")
            mc.label = label  # type: ignore[attr-defined]
            mc_samples_list.append(mc)
            corner_labels.append(label)

        if args.corner_include_true:
            for idx_true, (iface, label) in enumerate(zip(interfaces, labels)):
                true_path = iface.repository.root / "true_samples.csv"
                if not true_path.exists():
                    console.print(
                        f"[yellow]true_samples.csv not found for {label} at {true_path}[/yellow]"
                    )
                    continue
                try:
                    true_samples = _load_true_samples(true_path)
                except Exception as exc:
                    console.print(
                        f"[yellow]Failed to load true samples for {label}: {exc}[/yellow]"
                    )
                    continue
                missing_cols = [c for c in param_columns if c not in true_samples.data]
                if missing_cols:
                    console.print(
                        f"[yellow]Skipping true samples for {label}; missing columns {missing_cols}[/yellow]"
                    )
                    continue
                mc_true = true_samples.to_getdist(
                    param_columns=param_columns,
                    weight_column="weights",
                )
                true_label = (
                    true_corner_labels[idx_true]
                    if true_corner_labels is not None
                    else f"{label} true"
                )
                mc_true.label = true_label  # type: ignore[attr-defined]
                mc_samples_list.append(mc_true)
                corner_labels.append(true_label)

        plotter = plots.get_subplot_plotter()
        plotter.settings.progress = False
        triangle_kwargs: dict[str, object] = {}
        if args.corner_markers is not None:
            triangle_kwargs["markers"] = args.corner_markers
        contour_probs: list[float] | None = None
        if args.corner_sigmas is not None:
            sigma_array = np.asarray(args.corner_sigmas, dtype=np.float64)
            if sigma_array.size == 0:
                raise ValueError("--corner-sigmas requires at least one value.")
            if not np.all(np.isfinite(sigma_array)):
                raise ValueError("--corner-sigmas must contain finite values.")
            if np.any(sigma_array < 0):
                raise ValueError("--corner-sigmas must be non-negative.")
            sigma_sorted = np.sort(sigma_array)
            contour_probs = sigma_to_prob1D(sigma_sorted.tolist()).tolist()
            for mc in mc_samples_list:
                mc.updateSettings({"contours": contour_probs})
            plotter.settings.num_plot_contours = len(contour_probs)
        legend_labels = None if args.corner_no_legend else corner_labels
        triangle_kwargs_local = dict(triangle_kwargs)
        if legend_labels is not None:
            triangle_kwargs_local["legend_labels"] = legend_labels
        plotter.triangle_plot(
            mc_samples_list,
            params=param_columns,
            filled=filled,
            **triangle_kwargs_local,
        )
        if args.corner_simple_titles and mc_samples_list:
            _simplify_corner_titles(plotter, mc_samples_list[0], list(param_columns))
        corner_path = Path(args.corner).expanduser()
        corner_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.export(str(corner_path), dpi=300)
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
            color_cycle = SKY_PROBABILITY_COLOR_CYCLE
            cycle_len = len(color_cycle)
            plot_entries = list(zip(interfaces, samples_list, labels))
            total_runs = len(plot_entries)

            plt.figure(figsize=(5, 3))
            legend_handles = []
            from matplotlib.lines import Line2D

            if args.sky_top_quad_legacy:
                quad_mode = "legacy"
            elif args.sky_top_quad:
                quad_mode = "modern"
            else:
                quad_mode = "none"

            truth_pairs: list[tuple[float, float]] = []
            truth_labels: list[str] = []
            if args.sky_truth:
                if len(args.sky_truth) % 2 != 0:
                    raise ValueError("--sky-truth must be supplied as lon/lat pairs.")
                it = iter(args.sky_truth)
                truth_pairs = [(float(lon), float(lat)) for lon, lat in zip(it, it)]
                if args.sky_truth_labels:
                    if len(args.sky_truth_labels) != len(truth_pairs):
                        raise ValueError("--sky-truth-labels must match number of --sky-truth pairs.")
                    truth_labels = list(args.sky_truth_labels)
                else:
                    truth_labels = [f"truth {i+1}" for i in range(len(truth_pairs))]

            for idx, (iface, s, label) in enumerate(plot_entries):
                reverse_idx = (total_runs - idx - 1) % cycle_len
                color = color_cycle[reverse_idx]
                disable_mesh = idx > 0
                no_axes = idx > 0
                truth = truth_pairs if idx == 0 else None
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
                    top_quad=(quad_mode == "modern"),
                    top_quad_mode=quad_mode,
                    contour_levels=args.sky_contours,
                )
                # commented out to suppress prob contours
                # entry_label = label if idx == 0 else " "
                # legend_handles.append(Line2D([0], [0], color=color, lw=2, label=entry_label))

            for idx, truth_label in enumerate(truth_labels):
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        marker=marker_cycle[idx % len(marker_cycle)],
                        linestyle="None",
                        label=truth_label,
                    )
                )

            legend = None
            if legend_handles:
                legend_kwargs: dict[str, object] = {}
                if quad_mode == "modern":
                    legend_kwargs.update(
                        dict(
                            loc="center left",
                            bbox_to_anchor=(0.75, 0.85),
                            frameon=False,
                            borderaxespad=0.2,
                        )
                    )
                elif quad_mode == "legacy":
                    legend_kwargs["ncol"] = max(1, len(legend_handles))
                    legend_kwargs.update(
                        dict(loc="lower left", bbox_to_anchor=(0.46, 0.41))
                    )
                else:
                    legend_kwargs.update(dict(loc="lower right"))

                legend = plt.legend(handles=legend_handles, **legend_kwargs)
                if quad_mode == "modern":
                    legend.set_frame_on(False)

            sky_path = Path(args.sky_prob).expanduser()
            sky_path.parent.mkdir(parents=True, exist_ok=True)
            bbox_inches: str | Bbox
            if quad_mode in {"legacy", "modern"}:
                ax = plt.gca()
                fig = plt.gcf()
                bbox_inches = get_top_quadrant_bbox(ax, fig, plot_style=quad_mode)
                if quad_mode == "legacy":
                    x_labels, y_labels = quad_tick_labels()
                    ax.set_xticklabels(x_labels)
                    ax.set_yticklabels(y_labels)
                    ax.yaxis.tick_right()
                if legend is not None and quad_mode == "legacy":
                    legend.set_bbox_to_anchor((0.46, 0.41))
            else:
                bbox_inches = "tight"

            plt.savefig(sky_path, dpi=300, bbox_inches=bbox_inches)
            plt.close(plt.gcf())
            console.print(f"[green]Sky probability plot written to {sky_path}[/green]")
        except Exception as exc:
            plt.close(plt.gcf())
            console.print(f"[red]Failed to generate sky probability plot: {exc}[/red]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
