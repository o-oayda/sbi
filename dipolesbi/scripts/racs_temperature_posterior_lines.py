import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from anesthetic import read_csv as nested_read_csv
from catsim import RacsLow3, RacsLow3Config


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

_RUN_PATTERN = re.compile(r"samples_rnd-(\d+)\.csv$")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing samples_rnd-*.csv for a RACS SBI run.",
    )
    return parser


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
    required_columns = {"temp_slope", "temp_pivot_c", "logL"}
    missing = required_columns.difference(nested.columns)
    if missing:
        raise KeyError(f"Missing required posterior columns in {samples_path}: {sorted(missing)}")

    draws = nested.sample(n=POSTERIOR_SAMPLES, replace=True, random_state=SEED)
    slopes = np.asarray(draws["temp_slope"], dtype=np.float64)
    pivots = np.asarray(draws["temp_pivot_c"], dtype=np.float64)
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
) -> tuple[float, float, float]:
    if CENTRAL_LINE == "best_fit":
        index = int(np.nanargmax(logl))
        slope = float(slopes[index])
        pivot = float(pivots[index])
    elif CENTRAL_LINE == "median":
        slope = float(np.nanmedian(slopes))
        pivot = float(np.nanmedian(pivots))
    else:
        raise ValueError(f"Unsupported CENTRAL_LINE mode: {CENTRAL_LINE}")

    intercept = 1.0 - slope
    return slope, pivot, intercept


def _plot_lines(
    temperature_grid: np.ndarray,
    epsilon_lines: np.ndarray,
    run_dir: Path,
    round_id: int,
    central_line: np.ndarray,
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
        label=f"{CENTRAL_LINE.replace('_', ' ')} line",
    )
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel(r"$\epsilon(T)$")
    ax.set_title(f"{run_dir.name} posterior lines, round {round_id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


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
    )

    print(f"run_dir={run_dir}")
    print(f"round_id={round_id}")
    print(f"samples_path={samples_path}")
    print(f"posterior_samples={POSTERIOR_SAMPLES}")
    print(f"temperature_min={temperature_grid[0]:.6f}")
    print(f"temperature_max={temperature_grid[-1]:.6f}")
    print(f"central_line_mode={CENTRAL_LINE}")
    print(f"output_path={output_path}")


if __name__ == "__main__":
    main()
