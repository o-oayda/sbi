from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
import inspect
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np
from numpy.typing import NDArray
import healpy as hp
from anesthetic import read_csv as nested_read_csv
from getdist import MCSamples, plots
from matplotlib import pyplot as plt
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from dipolesbi.tools.configs import CatwiseConfig, ModelConfig
from dipolesbi.catwise.maps import Catwise
from dipolesbi.tools.utils import HidePrints, batch_simulate
from dipolesbi.tools.np_rngkey import NPKey
from dipolesbi.tools.plotting import sky_probability
from dipolesbi.tools.maps import average_smooth_map

_RUN_PATTERN = re.compile(r"samples_rnd-(\d+)\.csv$")


@dataclass(frozen=True)
class PosteriorRunInfo:
    """Metadata describing one posterior sample file."""

    round_id: int
    path: Path
    n_samples: int
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.n_samples < 0:
            msg = "n_samples must be non-negative"
            raise ValueError(msg)


@dataclass
class PosteriorSamples:
    """Concrete posterior samples for a given round."""

    info: PosteriorRunInfo
    data: Mapping[str, NDArray[np.float64]]

    def __getitem__(self, key: str) -> NDArray[np.float64]:
        return self.data[key]

    @property
    def columns(self) -> tuple[str, ...]:
        return self.info.columns

    @property
    def round_id(self) -> int:
        return self.info.round_id

    @property
    def n_samples(self) -> int:
        return self.info.n_samples

    def parameter_columns(self) -> list[str]:
        """Return the default set of columns treated as inference parameters."""
        return default_parameter_columns(self.columns)

    def to_getdist(
        self,
        *,
        param_columns: Sequence[str] | None = None,
        weight_column: str = "weights",
        name_map: Mapping[str, str] | None = None,
        label_map: Mapping[str, str] | None = None,
    ) -> MCSamples:
        param_cols = list(param_columns) if param_columns else self.parameter_columns()
        if not param_cols:
            msg = (
                "No parameter columns selected for GetDist conversion. "
                "Specify param_columns explicitly."
            )
            raise ValueError(msg)
        matrix = np.column_stack([np.asarray(self[col], dtype=np.float64) for col in param_cols])

        weights = np.asarray(self.data.get(weight_column, np.ones(self.n_samples)), dtype=np.float64)
        if weights.shape == ():
            weights = np.full(self.n_samples, float(weights))
        if weights.ndim != 1:
            weights = weights.reshape(-1)
        if weights.size != self.n_samples or not np.isfinite(weights).any():
            weights = np.ones(self.n_samples, dtype=np.float64)
        if np.allclose(weights.sum(), 0.0):
            weights = np.ones(self.n_samples, dtype=np.float64)

        names = [name_map[col] if name_map and col in name_map else col for col in param_cols]
        labels = [label_map[col] if label_map and col in label_map else format_label(col) for col in param_cols]

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            mc_samples = MCSamples(
                samples=matrix,
                weights=weights,
                sampler="nested",
                names=names,
                labels=labels,
                settings={"ignore_rows": 0.0},
            )
        return mc_samples

    def log_evidence(
        self,
        *,
        bootstrap: int | None = 100,
    ) -> tuple[float, float | None]:
        """Estimate log-evidence and its bootstrap uncertainty."""
        nested = nested_read_csv(self.info.path)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            logz_mean = float(nested.logZ())
            logz_err: float | None = None
            if bootstrap and bootstrap > 0:
                try:
                    logz_samples = np.asarray(nested.logZ(bootstrap), dtype=np.float64)
                    if logz_samples.size > 1:
                        logz_err = float(logz_samples.std())
                except Exception:
                    logz_err = None
        return logz_mean, logz_err


@dataclass
class PosteriorPredictiveResult:
    maps: NDArray[np.float32]
    masks: NDArray[np.bool_]
    sample_indices: NDArray[np.int64]
    output_paths: list[Path]
    downscale_nside: int | None


class PosteriorRepository:
    """Loads posterior samples produced by the multi-round pipelines."""

    def __init__(self, experiment_dir: Path | str) -> None:
        self._root = Path(experiment_dir).expanduser().resolve()
        if not self._root.is_dir():
            msg = f"Experiment directory {self._root} not found."
            raise FileNotFoundError(msg)
        self._runs = self._discover_runs()
        if not self._runs:
            msg = (
                f"No posterior CSV files matching samples_rnd-*.csv in {self._root}."
            )
            raise FileNotFoundError(msg)
        self._model_config: ModelConfig | None = None

    @property
    def root(self) -> Path:
        return self._root

    def available_rounds(self) -> tuple[int, ...]:
        return tuple(sorted(self._runs))

    def run_info(self, round_id: int) -> PosteriorRunInfo:
        try:
            return self._runs[round_id]
        except KeyError as exc:
            msg = f"Round {round_id} not available."
            raise KeyError(msg) from exc

    def iter_runs(self) -> Iterator[PosteriorRunInfo]:
        for round_id in self.available_rounds():
            yield self._runs[round_id]

    def load(self, round_id: int) -> PosteriorSamples:
        info = self.run_info(round_id)
        data = self._load_csv(info.path, info.columns)
        return PosteriorSamples(info=info, data=data)

    def model_config(self) -> ModelConfig:
        if self._model_config is None:
            self._model_config = self._load_model_config()
        return self._model_config

    def _discover_runs(self) -> dict[int, PosteriorRunInfo]:
        runs: dict[int, PosteriorRunInfo] = {}
        for path in sorted(self._root.glob("samples_rnd-*.csv")):
            match = _RUN_PATTERN.search(path.name)
            if not match:
                continue
            round_id = int(match.group(1))
            info = self._inspect_file(path, round_id)
            runs[round_id] = info
        return runs

    def _inspect_file(self, path: Path, round_id: int) -> PosteriorRunInfo:
        nested = nested_read_csv(path)
        base_columns = list(nested.columns)
        columns = ["sample_index", "weights", *base_columns]
        return PosteriorRunInfo(
            round_id=round_id,
            path=path,
            n_samples=len(nested),
            columns=tuple(columns),
        )

    def _load_csv(
        self,
        path: Path,
        expected_columns: Sequence[str],
    ) -> dict[str, NDArray[np.float64]]:
        nested = nested_read_csv(path)
        data: dict[str, NDArray[np.float64]] = {}
        data["sample_index"] = np.arange(len(nested), dtype=np.float64)
        weights = np.asarray(nested.get_weights(), dtype=np.float64)
        data["weights"] = weights
        for column in nested.columns:
            data[column] = np.asarray(nested[column], dtype=np.float64)
        missing = set(expected_columns) - set(data.keys())
        if missing:
            msg = f"Column mismatch when loading {path}: missing {sorted(missing)}."
            raise ValueError(msg)
        return data

    def _load_model_config(self) -> ModelConfig:
        config_path = self._root / "configs.txt"
        if not config_path.exists():
            msg = f"configs.txt not found in {self._root}"
            raise FileNotFoundError(msg)
        lines = config_path.read_text().splitlines()
        try:
            start_idx = next(
                idx for idx, line in enumerate(lines)
                if line.strip().startswith("ModelConfig")
            )
        except StopIteration as exc:
            raise ValueError(
                f"ModelConfig section not found in {config_path}"
            ) from exc

        config_line: str | None = None
        for line in lines[start_idx + 1:]:
            stripped = line.strip()
            if not stripped:
                continue
            config_line = stripped
            break

        if config_line is None:
            raise ValueError(f"ModelConfig entry is empty in {config_path}")

        namespace = {"CatwiseConfig": CatwiseConfig}
        try:
            model_config = eval(config_line, {"__builtins__": {}}, namespace)
        except Exception as exc:  # noqa: S307 (controlled namespace)
            raise ValueError(
                f"Failed to parse ModelConfig from {config_path}: {exc}"
            ) from exc

        if not isinstance(model_config, ModelConfig):
            msg = (
                f"Parsed ModelConfig has unexpected type: {type(model_config)}"
            )
            raise TypeError(msg)
        return model_config

_EXCLUDED_COLUMNS = {"weights", "sample_index", "nlive"}
_EXCLUDED_SUBSTRINGS = ("logl",)


def default_parameter_columns(columns: Sequence[str]) -> list[str]:
    """Heuristically filter out bookkeeping columns such as weights and logL."""
    selected: list[str] = []
    for col in columns:
        low = col.lower()
        if low in _EXCLUDED_COLUMNS:
            continue
        if any(substr in low for substr in _EXCLUDED_SUBSTRINGS):
            continue
        selected.append(col)
    return selected


def format_label(name: str) -> str:
    """Derive a simple LaTeX-friendly label for GetDist plots."""
    return name.replace("_", r"\_")


_CATWISE_PARAMETER_NAMES = [
    name
    for name, param in inspect.signature(Catwise.generate_dipole).parameters.items()
    if name not in {"self", "rng_key"}
]
_CATWISE_REQUIRED_PARAMETERS = {"log10_n_initial_samples"}


def _catwise_parameters_from_nested(
    draws,
) -> dict[str, NDArray[np.float32]]:
    params: dict[str, NDArray[np.float32]] = {}
    for name in _CATWISE_PARAMETER_NAMES:
        if name not in draws:
            continue
        values = np.asarray(draws[name], dtype=np.float64)
        params[name] = values.astype(np.float32, copy=False)
    missing_required = _CATWISE_REQUIRED_PARAMETERS - set(params)
    if missing_required:
        missing_str = ", ".join(sorted(missing_required))
        raise ValueError(
            f"Posterior samples do not provide required Catwise parameters: {missing_str}"
        )
    return params


def _plot_predictive_maps(
    maps: NDArray[np.float32],
    masks: NDArray[np.bool_],
    destinations: Sequence[Path],
) -> None:
    smoothed_maps: list[np.ndarray] = []
    for map_data, mask in zip(maps, masks, strict=True):
        mask_bool = np.asarray(mask, dtype=bool)
        m = np.asarray(map_data, dtype=np.float64).copy()
        m[~mask_bool] = np.nan
        smoothed = average_smooth_map(m)
        smoothed_maps.append(smoothed)

    for idx, (m, destination) in enumerate(zip(smoothed_maps, destinations, strict=True), start=1):
        finite_values = m[np.isfinite(m)]
        if finite_values.size:
            vmin = float(np.min(finite_values))
            vmax = float(np.max(finite_values))
            if np.isclose(vmin, vmax):
                vmin = vmax = None
        else:
            vmin = vmax = None

        destination.parent.mkdir(parents=True, exist_ok=True)
        figure = plt.figure(figsize=(8, 4))
        kwargs: dict[str, Any] = {
            "fig": figure.number,
            "sub": (1, 1, 1),
            "hold": True,
            "nest": True,
            "cbar": True,
            "title": f"Predictive sample {idx}",
        }
        if vmin is not None and vmax is not None:
            kwargs["min"] = vmin
            kwargs["max"] = vmax
        hp.projview(m, projection_type="mollweide", **kwargs)
        figure.savefig(str(destination), bbox_inches="tight", dpi=300)
        plt.close(figure)


def _resolve_ppc_paths(
    output_path: Path | str | None,
    default_dir: Path,
    round_id: int,
    count: int,
) -> list[Path]:
    if count <= 0:
        raise ValueError("count must be positive when resolving PPC output paths.")

    base_name = f"posterior_predictive_round-{round_id}"

    def _numbered(stem: str, suffix: str, idx: int) -> str:
        return f"{stem}_sample-{idx}{suffix}"

    if output_path is None:
        if count == 1:
            return [default_dir / f"{base_name}.pdf"]
        return [
            default_dir / f"{base_name}_sample-{i}.pdf"
            for i in range(1, count + 1)
        ]

    candidate = Path(output_path).expanduser()
    if candidate.is_dir():
        return [
            candidate / f"{base_name}_sample-{i}.pdf"
            for i in range(1, count + 1)
        ]

    suffix = candidate.suffix if candidate.suffix else ".pdf"
    stem = candidate.with_suffix("").name
    parent = candidate.parent if candidate.parent != Path("") else default_dir

    if count == 1:
        final_path = candidate if candidate.suffix else candidate.with_suffix(suffix)
        return [final_path]

    return [
        parent / _numbered(stem, suffix, i)
        for i in range(1, count + 1)
    ]


def _effective_worker_count(requested: int | None, count: int) -> int:
    if requested is not None and requested > 0:
        return requested
    cpu_total = os.cpu_count() or 1
    return max(1, min(cpu_total, count))


class PosteriorSamplesInterface:
    """Small helper for inspecting posterior rounds from the terminal."""

    def __init__(
        self,
        experiment_dir: Path | str,
        console: Console | None = None,
    ) -> None:
        self._console = console or Console()
        self._repository = PosteriorRepository(experiment_dir)
        self._catwise_models: dict[int | None, Catwise] = {}

    @property
    def repository(self) -> PosteriorRepository:
        return self._repository

    def show_available_rounds(self) -> None:
        table = Table(title="Posterior Rounds", expand=True)
        table.add_column("Round", justify="right")
        table.add_column("Samples", justify="right")
        table.add_column("File", overflow="fold")
        for info in self._repository.iter_runs():
            table.add_row(
                str(info.round_id),
                str(info.n_samples),
                info.path.name,
            )
        self._console.print(table)

    def prompt_round(self, default: int | None = None) -> int:
        available = self._repository.available_rounds()
        if not available:
            msg = f"No posterior samples found in {self._repository.root}"
            raise RuntimeError(msg)
        default_round = default if default in available else available[-1]
        choices = [str(idx) for idx in available]
        try:
            answer = Prompt.ask(
                "Select round to load",
                choices=choices,
                default=str(default_round),
                show_choices=True,
            )
        except EOFError:
            return default_round
        return int(answer)

    def load_round(
        self,
        round_id: int | None = None,
        *,
        show_table: bool = True,
    ) -> PosteriorSamples:
        if round_id is None:
            if show_table:
                self.show_available_rounds()
            round_id = self.prompt_round()
        return self._repository.load(round_id)

    def make_corner_plot(
        self,
        output_path: Path | str,
        round_id: int | None = None,
        *,
        columns: Sequence[str] | None = None,
        weight_column: str = "weights",
        markers: Sequence[float] | None = None,
        filled: bool = True,
        show_table: bool = True,
        **triangle_kwargs: Any,
    ) -> Path:
        samples = self.load_round(round_id=round_id, show_table=show_table)
        return save_corner_plot(
            samples,
            output_path,
            param_columns=columns,
            weight_column=weight_column,
            markers=markers,
            filled=filled,
            **triangle_kwargs,
        )

    def posterior_predictive(
        self,
        *,
        count: int,
        round_id: int | None = None,
        seed: int | None = None,
        workers: int | None = None,
        output_path: Path | str | None = None,
        show_table: bool = True,
        downscale_nside: int | None = None,
    ) -> PosteriorPredictiveResult:
        if count <= 0:
            raise ValueError("Posterior predictive sample count must be positive.")
        samples = self.load_round(round_id=round_id, show_table=show_table)
        model_config = self._repository.model_config()
        if not isinstance(model_config, CatwiseConfig):
            raise NotImplementedError(
                f"Posterior predictive checks currently support only CatwiseConfig, "
                f"received {type(model_config).__name__}."
            )
        model = self._get_catwise_model(model_config, downscale_nside)

        nested = nested_read_csv(samples.info.path)
        draws = nested.sample(n=count, replace=True, random_state=seed)
        indices = np.asarray(draws.index.get_level_values(0), dtype=np.int64)
        theta = _catwise_parameters_from_nested(draws)

        sim_key = NPKey.from_seed(seed) if seed is not None else None
        effective_workers = _effective_worker_count(workers, count)
        maps, masks = batch_simulate(
            theta,
            model.generate_dipole,
            n_workers=effective_workers,
            rng_key=sim_key,
        )
        maps_arr = np.atleast_2d(np.asarray(maps, dtype=np.float32))
        masks_arr = np.atleast_2d(np.asarray(masks, dtype=bool))

        destinations = _resolve_ppc_paths(
            output_path,
            self._repository.root,
            samples.round_id,
            count,
        )
        _plot_predictive_maps(maps_arr, masks_arr, destinations)

        return PosteriorPredictiveResult(
            maps=maps_arr,
            masks=masks_arr,
            sample_indices=indices,
            output_paths=destinations,
            downscale_nside=downscale_nside,
        )

    def _get_catwise_model(
        self,
        config: CatwiseConfig,
        downscale_override: int | None,
    ) -> Catwise:
        effective_downscale = downscale_override
        key = effective_downscale
        if key not in self._catwise_models:
            cfg = replace(config, downscale_nside=effective_downscale)
            with HidePrints():
                model = Catwise(cfg)
                model.initialise_data()
            self._catwise_models[key] = model
        return self._catwise_models[key]

    def plot_sky_probability(
        self,
        output_path: Path | str,
        *,
        round_id: int | None = None,
        lon_column: str = "dipole_longitude",
        lat_column: str = "dipole_latitude",
        nside: int = 64,
        smooth: float | None = 0.05,
        truth_deg: tuple[float, float] | None = None,
        show_table: bool = True,
    ) -> Path:
        samples = self.load_round(round_id=round_id, show_table=show_table)
        try:
            lon = np.asarray(samples[lon_column], dtype=np.float64)
            lat = np.asarray(samples[lat_column], dtype=np.float64)
        except KeyError as exc:
            raise KeyError(
                f"Required columns '{lon_column}' and/or '{lat_column}' missing in posterior samples."
            ) from exc

        finite_mask = np.isfinite(lon) & np.isfinite(lat)
        if not np.any(finite_mask):
            raise ValueError("No finite dipole longitude/latitude samples available for sky probability plot.")

        coords = np.column_stack([lon[finite_mask], lat[finite_mask]])

        truth_star = None
        if truth_deg is not None:
            lon_deg, lat_deg = truth_deg
            phi_rad = np.deg2rad(lon_deg)
            lat_rad = np.deg2rad(lat_deg)
            truth_star = [phi_rad, lat_rad]

        destination = Path(output_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)

        sky_probability(
            coords,
            lonlat=True,
            nside=nside,
            smooth=smooth,
            save_path=str(destination),
            show=False,
            truth_star=truth_star,
        )
        plt.close("all")
        return destination


def save_corner_plot(
    samples: PosteriorSamples,
    output_path: Path | str,
    *,
    param_columns: Sequence[str] | None = None,
    weight_column: str = "weights",
    markers: Sequence[float] | None = None,
    filled: bool = True,
    **triangle_kwargs: Any,
) -> Path:
    param_cols = list(param_columns) if param_columns is not None else samples.parameter_columns()
    mc_samples = samples.to_getdist(param_columns=param_cols, weight_column=weight_column)

    marker_list: list[float] | None = None
    if markers is not None:
        marker_list = list(markers)
        if len(marker_list) != len(param_cols):
            msg = (
                "Number of marker entries must match number of parameter columns "
                f"({len(param_cols)})."
            )
            raise ValueError(msg)

    plot_kwargs: dict[str, Any] = dict(triangle_kwargs)
    if "filled" not in plot_kwargs:
        plot_kwargs["filled"] = filled
    if marker_list is not None:
        plot_kwargs["markers"] = marker_list

    plotter = plots.get_subplot_plotter()
    plotter.triangle_plot([mc_samples], **plot_kwargs)

    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    plotter.export(str(destination))
    plt.close("all")
    return destination
