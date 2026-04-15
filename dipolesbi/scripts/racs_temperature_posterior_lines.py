import argparse
import os
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from anesthetic import read_csv as nested_read_csv
from astropy.table import Table
from catsim import RacsLow3, RacsLow3Config
from catsim.racs import LOW3_TEMPERATURE_EPSILON_FLOOR
from dipoleska.models.multipole import Multipole
from dipoleutils.utils.data_loader import DataLoader
from dipoleutils.utils.mask import Masker
from dipoleutils.utils.plotting import smooth_map
from dipoleutils.utils.samples import CatalogueToMap


ROUND_ID = None
POSTERIOR_SAMPLES = 2_000
SEED = 0
GRID_SIZE = 256
T_MIN = None
T_MAX = None
LINE_ALPHA = 0.05
LINE_COLOR = "tab:blue"
CENTRAL_LINE = "median"
CENTRAL_LINE_COLOR = "black"
CENTRAL_LINE_WIDTH = 2.5
FLUX_MIN = 2.0
NSIDE = 64
CHUNK_SIZE = 50_000
ALPHA_MEAN = 0.8
ALPHA_SIGMA = 0.2
FRACTIONAL_ERROR_FLUX_MIN_MJY = 10.0
DEFAULT_TEMP_PIVOT_C = 25.0
DEFAULT_VISUAL_FLUX_MIN_MJY = 15.0
DEFAULT_VISUAL_FLUX_MAX_MJY = 1000.0
ASKAP_UTC_OFFSET_HOURS = 8.0
DIPOLE_UTILS_ROOT = Path.home() / "Documents" / "dipole-utils"

_RUN_PATTERN = re.compile(r"samples_rnd-(\d+)\.csv$")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing samples_rnd-*.csv for a RACS SBI run.",
    )
    parser.add_argument(
        "--apply-visual-correction",
        action="store_true",
        help=(
            "Load the real RACS-low3 catalogue, remove the median temperature "
            "flux scaling at source level, and save before/after visualisations."
        ),
    )
    parser.add_argument(
        "--visual-flux-min",
        type=float,
        default=DEFAULT_VISUAL_FLUX_MIN_MJY,
        help="Flux cut in mJy applied when building the LOW3 visualisation maps.",
    )
    parser.add_argument(
        "--visual-flux-max",
        type=float,
        default=DEFAULT_VISUAL_FLUX_MAX_MJY,
        help="Maximum flux in mJy applied when building the LOW3 visualisation maps.",
    )
    parser.add_argument(
        "--visual-nside",
        type=int,
        default=NSIDE,
        help="HEALPix nside used for the LOW3 visualisation maps.",
    )
    parser.add_argument(
        "--central-line",
        choices=["median", "best_fit"],
        default=CENTRAL_LINE,
        help="How to choose the single posterior line used for the correction.",
    )
    parser.add_argument(
        "--fit-corrected-dipole",
        action="store_true",
        help=(
            "Fit a dipole to the corrected density map using the point-by-point "
            "likelihood and make a corner plot plus Galactic sky posterior."
        ),
    )
    return parser


def _load_low3_visual_helpers():
    scripts_root = DIPOLE_UTILS_ROOT
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))

    from scripts.low3_plot_helpers import plot_density_relationships
    from scripts.paf_temperature_lookup import get_mean_paf_temperatures_for_mjd

    return plot_density_relationships, get_mean_paf_temperatures_for_mjd


def _resolve_samples_path(run_dir: Path) -> tuple[int, Path]:
    matches: list[tuple[int, Path]] = []
    for path in run_dir.glob("samples_rnd-*.csv"):
        match = _RUN_PATTERN.match(path.name)
        if match is None:
            continue
        matches.append((int(match.group(1)), path))

    if not matches:
        raise FileNotFoundError(f"No samples_rnd-*.csv files found in {run_dir}")

    matches.sort(key=lambda item: item[0])

    if ROUND_ID is None:
        return matches[-1]

    for round_id, path in matches:
        if round_id == ROUND_ID:
            return round_id, path

    raise FileNotFoundError(f"Round {ROUND_ID} not found in {run_dir}")


def _build_temperature_grid() -> np.ndarray:
    config = RacsLow3Config(
        flux_min=FLUX_MIN,
        nside=NSIDE,
        chunk_size=CHUNK_SIZE,
        use_float32=False,
        downscale_nside=None,
        store_final_samples=False,
        alpha_mean=ALPHA_MEAN,
        alpha_sigma=ALPHA_SIGMA,
        fractional_error_flux_min_mjy=FRACTIONAL_ERROR_FLUX_MIN_MJY,
    )
    model = RacsLow3(config)
    model.initialise_data()

    if model.temperature_map is None:
        raise RuntimeError("RACS model did not initialise a temperature_map.")

    finite_temperature = np.asarray(model.temperature_map, dtype=np.float64)
    finite_temperature = finite_temperature[np.isfinite(finite_temperature)]
    if finite_temperature.size == 0:
        raise RuntimeError("No finite values found in the RACS temperature map.")

    t_min = float(np.min(finite_temperature)) if T_MIN is None else float(T_MIN)
    t_max = float(np.max(finite_temperature)) if T_MAX is None else float(T_MAX)
    if not t_min < t_max:
        raise ValueError(f"Temperature bounds must satisfy t_min < t_max, got {t_min}, {t_max}")
    return np.linspace(t_min, t_max, GRID_SIZE, dtype=np.float64)


def _draw_posterior_lines(samples_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nested = nested_read_csv(samples_path)
    required_columns = {"temp_slope", "logL"}
    missing = required_columns.difference(nested.columns)
    if missing:
        raise KeyError(f"Missing required posterior columns in {samples_path}: {sorted(missing)}")

    draws = nested.sample(n=POSTERIOR_SAMPLES, replace=True, random_state=SEED)
    slopes = np.asarray(draws["temp_slope"], dtype=np.float64)
    if "temp_pivot_c" in draws.columns:
        pivots = np.asarray(draws["temp_pivot_c"], dtype=np.float64)
    else:
        pivots = np.full(slopes.shape, DEFAULT_TEMP_PIVOT_C, dtype=np.float64)
    intercepts = 1.0 - slopes
    logl = np.asarray(draws["logL"], dtype=np.float64)
    return slopes, pivots, intercepts, logl


def _evaluate_lines(
    temperature_grid: np.ndarray,
    slopes: np.ndarray,
    pivots: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    return slopes[:, None] * (temperature_grid[None, :] / pivots[:, None]) + intercepts[:, None]


def _central_parameters(
    slopes: np.ndarray,
    pivots: np.ndarray,
    logl: np.ndarray,
    central_line_mode: str,
) -> tuple[float, float, float]:
    if central_line_mode == "best_fit":
        index = int(np.nanargmax(logl))
        slope = float(slopes[index])
        pivot = float(pivots[index])
    elif central_line_mode == "median":
        slope = float(np.nanmedian(slopes))
        pivot = float(np.nanmedian(pivots))
    else:
        raise ValueError(f"Unsupported central line mode: {central_line_mode}")

    intercept = 1.0 - slope
    return slope, pivot, intercept


def _plot_lines(
    temperature_grid: np.ndarray,
    epsilon_lines: np.ndarray,
    run_dir: Path,
    round_id: int,
    central_line: np.ndarray,
    central_line_mode: str,
) -> Path:
    output_path = run_dir / f"temp_line_posterior_r{round_id}.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    for line in epsilon_lines:
        ax.plot(
            temperature_grid,
            line,
            color=LINE_COLOR,
            alpha=LINE_ALPHA,
            linewidth=1.0,
        )

    ax.plot(
        temperature_grid,
        central_line,
        color=CENTRAL_LINE_COLOR,
        alpha=1.0,
        linewidth=CENTRAL_LINE_WIDTH,
        label=f"{central_line_mode.replace('_', ' ')} line",
    )
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel(r"$\epsilon(T)$")
    ax.set_title(f"{run_dir.name} posterior lines, round {round_id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def _evaluate_temperature_enhancement(
    temperature_c: np.ndarray,
    slope: float,
    pivot_c: float,
    intercept: float,
) -> np.ndarray:
    enhancement = np.full(temperature_c.shape, intercept, dtype=np.float64)
    finite = np.isfinite(temperature_c)
    if np.any(finite):
        enhancement[finite] = slope * (temperature_c[finite] / pivot_c) + intercept
    return np.maximum(enhancement, LOW3_TEMPERATURE_EPSILON_FLOOR)


def _load_low3_catalogue_with_temperature() -> Table:
    _, get_mean_paf_temperatures_for_mjd = _load_low3_visual_helpers()
    catalogue = DataLoader("racs", "low3").load()
    catalogue = catalogue.copy()

    catalogue["Start_time_hours"] = np.mod(
        np.asarray(catalogue["Scan_start_MJD"], dtype=float) % 1.0 * 24.0
        + ASKAP_UTC_OFFSET_HOURS,
        24.0,
    )
    try:
        catalogue["Temperature_C"] = get_mean_paf_temperatures_for_mjd(
            catalogue["Scan_start_MJD"]
        )
    except Exception as exc:
        warnings.warn(f"Unable to fetch PAF temperature data: {exc}")
        catalogue["Temperature_C"] = np.full(len(catalogue), np.nan, dtype=np.float64)

    return catalogue


def _build_corrected_catalogue(
    catalogue: Table,
    slope: float,
    pivot_c: float,
    intercept: float,
) -> Table:
    corrected = catalogue.copy()
    source_temperature = np.asarray(corrected["Temperature_C"], dtype=np.float64)
    original_flux = np.asarray(corrected["Total_flux"], dtype=np.float64)
    enhancement = _evaluate_temperature_enhancement(
        source_temperature,
        slope=slope,
        pivot_c=pivot_c,
        intercept=intercept,
    )
    corrected_flux = original_flux.copy()
    finite_flux = np.isfinite(original_flux)
    finite_temperature = np.isfinite(source_temperature)
    corrected_rows = finite_flux & finite_temperature
    corrected_flux[corrected_rows] = original_flux[corrected_rows] / enhancement[corrected_rows]

    corrected["temperature_enhancement"] = enhancement
    corrected["Total_flux_temperature_corrected"] = corrected_flux
    return corrected


def _build_visual_maps(
    catalogue: Table,
    flux_column: str,
    flux_min: float,
    flux_max: float | None,
    nside: int,
) -> dict[str, np.ndarray]:
    processor = CatalogueToMap(catalogue.copy())
    processor.make_cut(flux_column, flux_min, flux_max)
    cut_catalogue = processor.get_catalogue()

    flux = np.asarray(cut_catalogue[flux_column], dtype=np.float64)
    flux_safe = np.clip(flux, np.finfo(np.float64).tiny, None)
    cut_catalogue["perc_err"] = (
        np.asarray(cut_catalogue["E_Total_flux"], dtype=np.float64) / flux_safe
    ) * 100.0

    dmap = processor.make_density_map("equatorial", nside=nside)
    fmap = processor.make_parameter_map(
        column_name=flux_column,
        coordinate_system="equatorial",
        operation="mean",
        nside=nside,
    )
    start_time_map = processor.make_parameter_map(
        column_name="Start_time_hours",
        coordinate_system="equatorial",
        operation="mean",
        nside=nside,
    )
    temperature_map = processor.make_parameter_map(
        column_name="Temperature_C",
        coordinate_system="equatorial",
        operation="mean",
        nside=nside,
    )
    pmap = processor.make_parameter_map(
        column_name="perc_err",
        coordinate_system="equatorial",
        operation="mean",
        nside=nside,
    )
    rms_map = processor.make_parameter_map(
        column_name="Noise",
        coordinate_system="equatorial",
        operation="mean",
        nside=nside,
    )
    psf_map = processor.make_parameter_map(
        column_name="PSF_Maj",
        coordinate_system="equatorial",
        operation="mean",
        nside=nside,
    )

    masker = Masker(
        [dmap, fmap, start_time_map, temperature_map, pmap, rms_map, psf_map],
        "equatorial",
    )
    masker.mask_galactic_plane(5)
    masker.mask_a_team_sources(radius_deg=3, source_names=["Cygnus A"])
    masker.mask_a_team_sources(radius_deg=2)
    masker.mask_equatorial_poles(north_radius=42)
    (
        masked_density_map,
        masked_flux_map,
        masked_start_time_map,
        masked_temperature_map,
        masked_percent_error_map,
        masked_rms_map,
        masked_psf_map,
    ) = masker.get_masked_density_map()

    return {
        "density_map": np.asarray(masked_density_map, dtype=np.float64),
        "flux_map": np.asarray(masked_flux_map, dtype=np.float64),
        "start_time_map": np.asarray(masked_start_time_map, dtype=np.float64),
        "temperature_map": np.asarray(masked_temperature_map, dtype=np.float64),
        "percent_error_map": np.asarray(masked_percent_error_map, dtype=np.float64),
        "rms_map": np.asarray(masked_rms_map, dtype=np.float64),
        "psf_map": np.asarray(masked_psf_map, dtype=np.float64),
        "time_hours": np.asarray(cut_catalogue["Start_time_hours"], dtype=np.float64),
        "source_temperatures": np.asarray(cut_catalogue["Temperature_C"], dtype=np.float64),
    }


def _save_density_relationship_plot(
    output_path: Path,
    density_map: np.ndarray,
    start_time_map: np.ndarray,
    temperature_map: np.ndarray,
    time_hours: np.ndarray,
    source_temperatures: np.ndarray,
    correction_temperature: np.ndarray,
    correction_factor: np.ndarray,
    title_prefix: str = "",
) -> Path:
    plot_density_relationships, _ = _load_low3_visual_helpers()
    fig, _ = plot_density_relationships(
        density_map,
        start_time_map,
        temperature_map,
        time_hours,
        source_temperatures,
        time_window_hours=2.0,
        temperature_window_c=3.0,
        temperature_time_bin_hours=0.25,
        title_prefix=title_prefix,
        correction_temperature=correction_temperature,
        correction_factor=correction_factor,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_smoothed_map(output_path: Path, density_map: np.ndarray, title: str) -> Path:
    plt.figure(figsize=(9, 6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        smooth_map(
            density_map,
            title=title,
            unit="sources per pixel",
            cmap="viridis",
        )
    plt.gcf().savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(plt.gcf())
    return output_path


def _save_smoothed_map_galactic(output_path: Path, density_map: np.ndarray, title: str) -> Path:
    plt.figure(figsize=(9, 6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        smooth_map(
            density_map,
            title=title,
            unit="sources per pixel",
            cmap="viridis",
            coord=["C", "G"],
        )
    plt.gcf().savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(plt.gcf())
    return output_path


def _save_density_difference_map(
    output_path: Path,
    corrected_density_map: np.ndarray,
    original_density_map: np.ndarray,
) -> Path:
    difference_map = corrected_density_map - original_density_map
    plt.figure(figsize=(9, 6))
    hp.projview(
        difference_map,
        title="Corrected - uncorrected LOW3 density map",
        unit="sources per pixel",
        cb_orientation="vertical",
        cmap="coolwarm",
    )
    plt.gcf().savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(plt.gcf())
    return output_path


def _save_density_difference_map_galactic(
    output_path: Path,
    corrected_density_map: np.ndarray,
    original_density_map: np.ndarray,
) -> Path:
    difference_map = corrected_density_map - original_density_map
    plt.figure(figsize=(9, 6))
    hp.projview(
        difference_map,
        title="Corrected - uncorrected LOW3 density map (Galactic)",
        unit="sources per pixel",
        cb_orientation="vertical",
        cmap="coolwarm",
        coord=["C", "G"],
    )
    plt.gcf().savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(plt.gcf())
    return output_path


def _run_visual_correction(
    run_dir: Path,
    slope: float,
    pivot_c: float,
    intercept: float,
    temperature_grid: np.ndarray,
    central_line: np.ndarray,
    flux_min: float,
    flux_max: float | None,
    nside: int,
) -> dict[str, Path]:
    catalogue = _load_low3_catalogue_with_temperature()
    corrected_catalogue = _build_corrected_catalogue(
        catalogue,
        slope=slope,
        pivot_c=pivot_c,
        intercept=intercept,
    )

    original_maps = _build_visual_maps(
        catalogue,
        flux_column="Total_flux",
        flux_min=flux_min,
        flux_max=flux_max,
        nside=nside,
    )
    corrected_maps = _build_visual_maps(
        corrected_catalogue,
        flux_column="Total_flux_temperature_corrected",
        flux_min=flux_min,
        flux_max=flux_max,
        nside=nside,
    )

    correction_factor = 1.0 / np.clip(
        np.asarray(central_line, dtype=np.float64),
        LOW3_TEMPERATURE_EPSILON_FLOOR,
        None,
    )

    outputs = {
        "original_diagnostics": _save_density_relationship_plot(
            run_dir / "racs_low3_temperature_diagnostics_uncorrected.png",
            original_maps["density_map"],
            original_maps["start_time_map"],
            original_maps["temperature_map"],
            original_maps["time_hours"],
            original_maps["source_temperatures"],
            temperature_grid,
            correction_factor,
        ),
        "corrected_diagnostics": _save_density_relationship_plot(
            run_dir / "racs_low3_temperature_diagnostics_corrected.png",
            corrected_maps["density_map"],
            corrected_maps["start_time_map"],
            corrected_maps["temperature_map"],
            corrected_maps["time_hours"],
            corrected_maps["source_temperatures"],
            temperature_grid,
            correction_factor,
            title_prefix="Corrected ",
        ),
        "original_sky_map": _save_smoothed_map(
            run_dir / "racs_low3_density_uncorrected.png",
            original_maps["density_map"],
            "Uncorrected RACS-low3 density map",
        ),
        "original_sky_map_galactic": _save_smoothed_map_galactic(
            run_dir / "racs_low3_density_uncorrected_galactic.png",
            original_maps["density_map"],
            "Uncorrected RACS-low3 density map (Galactic)",
        ),
        "corrected_sky_map": _save_smoothed_map(
            run_dir / "racs_low3_density_corrected.png",
            corrected_maps["density_map"],
            "Temperature-corrected RACS-low3 density map",
        ),
        "corrected_sky_map_galactic": _save_smoothed_map_galactic(
            run_dir / "racs_low3_density_corrected_galactic.png",
            corrected_maps["density_map"],
            "Temperature-corrected RACS-low3 density map (Galactic)",
        ),
        "difference_sky_map": _save_density_difference_map(
            run_dir / "racs_low3_density_difference.png",
            corrected_maps["density_map"],
            original_maps["density_map"],
        ),
        "difference_sky_map_galactic": _save_density_difference_map_galactic(
            run_dir / "racs_low3_density_difference_galactic.png",
            corrected_maps["density_map"],
            original_maps["density_map"],
        ),
    }
    return outputs


def _fit_corrected_dipole_and_plot(corrected_density_map: np.ndarray):
    model = Multipole(
        density_map=np.asarray(corrected_density_map, dtype=np.float64),
        likelihood="point",
        ells=[1,2]
    )
    model.run_nested_sampling(step=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
            category=UserWarning,
        )
        model.corner_plot(coordinates=["equatorial", "galactic"])
        model.sky_direction_posterior(coordinates=["equatorial", "galactic"])
        plt.show()
    return model


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    round_id, samples_path = _resolve_samples_path(run_dir)
    temperature_grid = _build_temperature_grid()
    slopes, pivots, intercepts, logl = _draw_posterior_lines(samples_path)
    epsilon_lines = _evaluate_lines(temperature_grid, slopes, pivots, intercepts)

    central_slope, central_pivot, central_intercept = _central_parameters(
        slopes,
        pivots,
        logl,
        args.central_line,
    )
    central_line = _evaluate_lines(
        temperature_grid,
        np.asarray([central_slope]),
        np.asarray([central_pivot]),
        np.asarray([central_intercept]),
    )[0]

    output_path = _plot_lines(
        temperature_grid,
        epsilon_lines,
        run_dir,
        round_id,
        central_line,
        args.central_line,
    )

    visual_outputs: dict[str, Path] = {}
    corrected_density_map: np.ndarray | None = None
    if args.apply_visual_correction:
        visual_outputs = _run_visual_correction(
            run_dir=run_dir,
            slope=central_slope,
            pivot_c=central_pivot,
            intercept=central_intercept,
            temperature_grid=temperature_grid,
            central_line=central_line,
            flux_min=float(args.visual_flux_min),
            flux_max=None if args.visual_flux_max is None else float(args.visual_flux_max),
            nside=int(args.visual_nside),
        )
        corrected_catalogue = _build_corrected_catalogue(
            _load_low3_catalogue_with_temperature(),
            slope=central_slope,
            pivot_c=central_pivot,
            intercept=central_intercept,
        )
        corrected_maps = _build_visual_maps(
            corrected_catalogue,
            flux_column="Total_flux_temperature_corrected",
            flux_min=float(args.visual_flux_min),
            flux_max=None if args.visual_flux_max is None else float(args.visual_flux_max),
            nside=int(args.visual_nside),
        )
        corrected_density_map = corrected_maps["density_map"]

    dipole_model: Dipole | None = None
    if args.fit_corrected_dipole:
        if corrected_density_map is None:
            corrected_catalogue = _build_corrected_catalogue(
                _load_low3_catalogue_with_temperature(),
                slope=central_slope,
                pivot_c=central_pivot,
                intercept=central_intercept,
            )
            corrected_maps = _build_visual_maps(
                corrected_catalogue,
                flux_column="Total_flux_temperature_corrected",
                flux_min=float(args.visual_flux_min),
                flux_max=None if args.visual_flux_max is None else float(args.visual_flux_max),
                nside=int(args.visual_nside),
            )
            corrected_density_map = corrected_maps["density_map"]
        dipole_model = _fit_corrected_dipole_and_plot(corrected_density_map)

    print(f"run_dir={run_dir}")
    print(f"round_id={round_id}")
    print(f"samples_path={samples_path}")
    print(f"posterior_samples={POSTERIOR_SAMPLES}")
    print(f"temperature_min={temperature_grid[0]:.6f}")
    print(f"temperature_max={temperature_grid[-1]:.6f}")
    print(f"central_line_mode={args.central_line}")
    print(f"central_slope={central_slope:.6f}")
    print(f"central_pivot_c={central_pivot:.6f}")
    print(f"central_intercept={central_intercept:.6f}")
    print(f"posterior_line_output_path={output_path}")
    if args.apply_visual_correction:
        print(f"visual_flux_min_mjy={float(args.visual_flux_min):.6f}")
        print(f"visual_flux_max_mjy={float(args.visual_flux_max):.6f}")
        print(f"visual_nside={int(args.visual_nside)}")
        for key, path in visual_outputs.items():
            print(f"{key}_output_path={path}")
    if dipole_model is not None:
        print("dipole_fit_likelihood=point")
        print(f"dipole_fit_n_samples={len(dipole_model.samples)}")


if __name__ == "__main__":
    main()
