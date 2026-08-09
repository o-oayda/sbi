import argparse
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from catsim import RACS_PRODUCTS, TEMPERATURE_MODELS, Racs, RacsConfig
from catsim.racs_jax import RacsJax
import healpy as hp
import jax
import numpy as np
import yaml

from dipolesbi.pipelines.racs_observation_helpers import (
    DEFAULT_FLUX_ELEVATION_N_BINS,
    DEFAULT_FLUX_ELEVATION_QUANTILES,
    DEFAULT_FLUX_TEMPERATURE_JAX_FLUX_BINS,
    DEFAULT_FLUX_TEMPERATURE_N_BINS,
    DEFAULT_FLUX_TEMPERATURE_QUANTILES,
    SummaryFeature,
    _model_product,
    build_hybrid_sample_from_native,
    build_mask,
    build_mask_from_observation_config,
    build_real_sample,
    load_catalogue,
    load_observation_config,
    load_reference_observation,
)
from dipolesbi.pipelines.summary_stats import (
    _flux_elevation_edges,
    _flux_elevation_quantile_ndim,
    _flux_temperature_edges,
    _flux_temperature_quantile_ndim,
    _native_count_log_dispersion_feature,
)
from dipolesbi.tools.configs import (
    DataTransformSpec,
    EmbeddingNetConfig,
    Scenario,
    ThetaTransformSpec,
)
from dipolesbi.tools.multiround_inferer import MultiRoundInferer
from dipolesbi.tools.np_rngkey import NPKey
from dipolesbi.tools.priors_np import DipolePriorNP
from dipolesbi.tools.summary_diagnostics import (
    QuantileSummaryDiagnosticSpec,
    make_round_quantile_diagnostic,
)
from dipolesbi.tools.ui import MultiRoundInfererUI
from dipolesbi.tools.utils import batch_simulate

def _write_run_command(out_dir: str) -> Path:
    """Write a shell-safe reconstruction of the Python invocation."""
    output_dir = Path(out_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    command_path = output_dir / "run_command.txt"
    argv = getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv]
    command_path.write_text(f"{shlex.join(argv)}\n", encoding="utf-8")
    return command_path


def load_inference_config(config_path: str | Path) -> dict[str, Any]:
    """Load a structured inference configuration from YAML."""
    path = Path(config_path).expanduser()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Inference config must contain a YAML mapping: {path}")
    return config


def construct_argparser() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        nargs="+",
        help=(
            "Optional compatibility check; when supplied, it must match the "
            "single mode declared by --inference_config."
        ),
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        required=True,
        help="Number of simulations to run.",
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=15,
        help="Number of rounds of inference.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of local workers to use for NumPy simulation batches.",
    )
    parser.add_argument(
        "--use_jax",
        action="store_true",
        help="Use catsim's JAX batched RACS simulator.",
    )
    parser.add_argument(
        "--jax_batch_size",
        type=int,
        default=5,
        help="Batch size passed to RacsJax.batch_generate_dipole.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Exact directory in which to save outputs.",
    )
    parser.add_argument(
        "--ssnle_seed",
        type=int,
        default=0,
        help="Seed used for sequential neural estimators.",
    )
    parser.add_argument(
        "--downscale_nside",
        type=int,
        default=None,
        help="Optional HEALPix nside to downscale maps to.",
    )
    parser.add_argument(
        "--no_ui",
        action="store_true",
        help="Disable the Rich multi-round progress UI.",
    )
    parser.add_argument(
        "--flux_min",
        type=float,
        default=2.0,
        help="RACS flux threshold in mJy.",
    )
    parser.add_argument(
        "--flux_temperature_min_mjy",
        type=float,
        default=None,
        help=(
            "Independent minimum flux for temperature-binned flux quantiles. "
            "Defaults to --flux_min."
        ),
    )
    parser.add_argument(
        "--racs_epoch",
        choices=sorted(RACS_PRODUCTS),
        default="low3",
        help="RACS product/epoch to simulate.",
    )
    parser.add_argument(
        "--catalogue_path",
        type=Path,
        required=True,
        help="Path to the exact raw RACS catalogue used by the simulator and observation.",
    )
    parser.add_argument(
        "--observation_config",
        type=Path,
        required=True,
        help=(
            "Observation YAML used to prepare --reference_observation and to "
            "define the simulator's native-resolution mask."
        ),
    )
    parser.add_argument(
        "--inference_config",
        type=Path,
        required=True,
        help="YAML defining the NLE or NPE scenario, flow, and training setup.",
    )
    parser.add_argument(
        "--reference_observation",
        type=Path,
        required=True,
        help="Prepared NPZ containing the final x0 data vector and mask.",
    )
    parser.add_argument(
        "--local_source_crossmatch_radius_arcsec",
        type=float,
        default=None,
        help=(
            "Remove sources matched to the local-source catalogue within this "
            "radius in arcseconds. Omit to disable local-source cross-matching."
        ),
    )
    parser.add_argument(
        "--temperature_fallback",
        choices=("none", "open_meteo", "reference"),
        default="none",
        help="Strategy used when one or more PAF temperatures are unavailable.",
    )
    parser.add_argument(
        "--paf_reference_temp_c",
        type=float,
        help=(
            "Explicit reference temperature used by --temperature_fallback "
            "reference."
        ),
    )
    parser.add_argument(
        "--max_reference_fallback_tiles",
        type=int,
        help=(
            "Maximum number of missing tiles that may use the explicit "
            "reference temperature."
        ),
    )
    parser.add_argument(
        "--paf_temperature_data_dir",
        type=str,
        help=(
            "Directory containing PAF temperature data. May be omitted for "
            "LOW2 when --temperature_fallback open_meteo is selected."
        ),
    )
    parser.add_argument(
        "--noisemap_data_dir",
        type=str,
        required=True,
        help="Directory containing the product-specific RACS noise map.",
    )
    parser.add_argument(
        "--temperature_model",
        choices=sorted(TEMPERATURE_MODELS),
        default="hot_linear",
        help="Functional form of the hot-temperature flux response.",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=64,
        help="Native HEALPix nside used by the simulator.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help=(
            "Chunk size used inside the simulator. Defaults to 2_500_000 for "
            "NumPy and 140_000 for JAX."
        ),
    )
    parser.add_argument(
        "--alpha_mean",
        type=float,
        default=0.8,
        help="Mean of the Gaussian spectral-index model.",
    )
    parser.add_argument(
        "--alpha_sigma",
        type=float,
        default=0.2,
        help="Standard deviation of the Gaussian spectral-index model.",
    )
    parser.add_argument(
        "--noise_map_nside",
        type=int,
        default=256,
        help="HEALPix nside of the cached RACS noise map.",
    )
    parser.add_argument(
        "--flux_error_noise_bins",
        type=int,
        default=None,
        help=(
            "Number of log-noise bins in the conditional flux-error lookup. "
            "Defaults to the selected RACS product value."
        ),
    )
    parser.add_argument(
        "--flux_error_flux_bins",
        type=int,
        default=None,
        help=(
            "Number of log-flux bins in the conditional flux-error lookup. "
            "Defaults to the selected RACS product value."
        ),
    )
    parser.add_argument(
        "--flux_error_min_cell_count",
        type=int,
        default=10,
        help="Minimum catalogue occupancy for a directly sampled lookup cell.",
    )
    parser.add_argument(
        "--flux_error_noise_bounds_ujy_beam",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Noise-axis lookup bounds; defaults to the selected product value.",
    )
    parser.add_argument(
        "--flux_error_flux_bounds_mjy",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Flux-axis lookup bounds; defaults to the selected product value.",
    )
    parser.add_argument(
        "--summary_features",
        nargs="*",
        choices=["log_dispersion", "flux_quantiles", "flux_elevation_quantiles"],
        default=[],
        help="Summary features to append after the map for NLE.",
    )
    parser.add_argument(
        "--flux_temperature_n_bins",
        type=int,
        default=DEFAULT_FLUX_TEMPERATURE_N_BINS,
        help="Number of temperature bins used by the flux-quantile summary.",
    )
    parser.add_argument(
        "--flux_temperature_quantiles",
        type=float,
        nargs="+",
        default=DEFAULT_FLUX_TEMPERATURE_QUANTILES,
        help="Flux quantiles to compute within each temperature bin.",
    )
    parser.add_argument(
        "--flux_elevation_n_bins",
        type=int,
        default=DEFAULT_FLUX_ELEVATION_N_BINS,
        help="Number of elevation bins used by the elevation flux-quantile summary.",
    )
    parser.add_argument(
        "--flux_elevation_quantiles",
        type=float,
        nargs="+",
        default=DEFAULT_FLUX_ELEVATION_QUANTILES,
        help="Flux quantiles to compute within each elevation bin.",
    )
    parser.add_argument(
        "--simulate_clustering",
        type=str,
        choices=["geometric", "poisson"],
        default=None,
        help="Use a clustering model to represent extended sources.",
    )
    parser.add_argument(
        "--add_extra_error",
        action='store_true',
        help="Add parameter for extra error on flux."
    )
    parser.add_argument("--log10_n_initial_samples_min", type=float, default=5.6)
    parser.add_argument("--log10_n_initial_samples_max", type=float, default=6.8)
    parser.add_argument("--observer_speed_min", type=float, default=0.0)
    parser.add_argument("--observer_speed_max", type=float, default=12.0)
    parser.add_argument("--dipole_longitude_min", type=float, default=0.0)
    parser.add_argument("--dipole_longitude_max", type=float, default=360.0)
    parser.add_argument("--dipole_latitude_min", type=float, default=-90.0)
    parser.add_argument("--dipole_latitude_max", type=float, default=90.0)
    parser.add_argument("--temp_beta_min", type=float, default=0.0)
    parser.add_argument("--temp_beta_max", type=float, default=0.05)
    parser.add_argument(
        "--add_elevation_model",
        action="store_true",
        help="Infer the RACS elevation sensitivity model parameters.",
    )
    parser.add_argument("--elevation_trough_min", type=float, default=0.0)
    parser.add_argument("--elevation_trough_max", type=float, default=90.0)
    parser.add_argument("--elevation_amp_min", type=float, default=0.0)
    parser.add_argument("--elevation_amp_max", type=float, default=0.2)
    parser.add_argument("--p_clus_min", type=float, default=0.0)
    parser.add_argument("--p_clus_max", type=float, default=1.0)
    parser.add_argument("--clus_stop_prob_min", type=float, default=0.4)
    parser.add_argument("--clus_stop_prob_max", type=float, default=1.0)
    parser.add_argument("--lambda_clus_min", type=float, default=0.0)
    parser.add_argument("--lambda_clus_max", type=float, default=3.0)
    parser.add_argument("--eta_min", type=float, default=0.0)
    parser.add_argument("--eta_max", type=float, default=8.0)
    parser.add_argument("--max_children", type=int, default=16)
    return parser.parse_args(), parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    if args.chunk_size is None:
        args.chunk_size = 140_000 if args.use_jax else 2_500_000
    if args.paf_temperature_data_dir is None and not (
        args.racs_epoch == "low2" and args.temperature_fallback == "open_meteo"
    ):
        parser.error(
            "--paf_temperature_data_dir may only be omitted for LOW2 when "
            "--temperature_fallback open_meteo is selected."
        )
    if args.temperature_fallback == "reference":
        if args.paf_reference_temp_c is None or not np.isfinite(
            args.paf_reference_temp_c
        ):
            parser.error(
                "--paf_reference_temp_c must be explicitly set to a finite "
                "value for reference fallback."
            )
        if (
            args.max_reference_fallback_tiles is None
            or args.max_reference_fallback_tiles <= 0
        ):
            parser.error(
                "--max_reference_fallback_tiles must be explicitly set to a "
                "positive integer for reference fallback."
            )
    elif (
        args.paf_reference_temp_c is not None
        or args.max_reference_fallback_tiles is not None
    ):
        parser.error(
            "--paf_reference_temp_c and --max_reference_fallback_tiles may only "
            "be used with --temperature_fallback reference."
        )
    if args.use_jax and args.n_workers is not None:
        parser.error("--n_workers is only used by the NumPy simulator.")
    if args.use_jax and args.jax_batch_size <= 0:
        parser.error("--jax_batch_size must be positive.")
    if not np.isfinite(args.flux_min) or args.flux_min <= 0.0:
        parser.error("--flux_min must be positive and finite.")
    if args.local_source_crossmatch_radius_arcsec is not None and (
        not np.isfinite(args.local_source_crossmatch_radius_arcsec)
        or args.local_source_crossmatch_radius_arcsec <= 0.0
    ):
        parser.error(
            "--local_source_crossmatch_radius_arcsec must be positive and finite."
        )
    if args.flux_temperature_min_mjy is not None and (
        not np.isfinite(args.flux_temperature_min_mjy)
        or args.flux_temperature_min_mjy <= 0.0
    ):
        parser.error("--flux_temperature_min_mjy must be positive and finite.")
    if args.flux_temperature_n_bins < 1:
        parser.error("--flux_temperature_n_bins must be at least 1.")
    if not args.flux_temperature_quantiles:
        parser.error("--flux_temperature_quantiles must contain at least one value.")
    if any(q < 0.0 or q > 1.0 for q in args.flux_temperature_quantiles):
        parser.error("--flux_temperature_quantiles values must lie in [0, 1].")
    if args.flux_elevation_n_bins < 1:
        parser.error("--flux_elevation_n_bins must be at least 1.")
    if not args.flux_elevation_quantiles:
        parser.error("--flux_elevation_quantiles must contain at least one value.")
    if any(q < 0.0 or q > 1.0 for q in args.flux_elevation_quantiles):
        parser.error("--flux_elevation_quantiles values must lie in [0, 1].")
    if "flux_elevation_quantiles" in args.summary_features:
        if args.racs_epoch != "mid1":
            parser.error("Flux-elevation summaries require --racs_epoch mid1.")
    if (
        args.noise_map_nside <= 0
        or args.noise_map_nside & (args.noise_map_nside - 1)
    ):
        parser.error("--noise_map_nside must be a positive power of two.")
    product = RACS_PRODUCTS[args.racs_epoch]
    if args.flux_error_noise_bins is None:
        args.flux_error_noise_bins = product.default_flux_error_noise_bins
    if args.flux_error_flux_bins is None:
        args.flux_error_flux_bins = product.default_flux_error_flux_bins
    if args.flux_error_noise_bounds_ujy_beam is None:
        args.flux_error_noise_bounds_ujy_beam = (
            product.default_flux_error_noise_bounds_ujy_beam
        )
    if args.flux_error_flux_bounds_mjy is None:
        args.flux_error_flux_bounds_mjy = (
            product.default_flux_error_flux_bounds_mjy
        )
    if args.flux_error_noise_bins < 2:
        parser.error("--flux_error_noise_bins must be at least 2.")
    if args.flux_error_flux_bins < 2:
        parser.error("--flux_error_flux_bins must be at least 2.")
    if args.flux_error_min_cell_count < 1:
        parser.error("--flux_error_min_cell_count must be at least 1.")
    for name in (
        "flux_error_noise_bounds_ujy_beam",
        "flux_error_flux_bounds_mjy",
    ):
        bounds = getattr(args, name)
        if (
            bounds is None
            or len(bounds) != 2
            or not np.all(np.isfinite(bounds))
            or bounds[0] <= 0
            or bounds[1] <= bounds[0]
        ):
            parser.error(
                f"--{name} must contain two finite positive values in "
                "strictly increasing order."
            )
    _validate_prior_bounds(args, parser)

    modes = _parse_modes(args.mode, parser)
    if args.summary_features and any(mode != "NLE" for mode in modes):
        parser.error("Summary statistics are only supported for the NLE at the moment.")

    args.flux_temperature_quantiles = tuple(args.flux_temperature_quantiles)
    args.flux_elevation_quantiles = tuple(args.flux_elevation_quantiles)
    return modes


def _validate_prior_bounds(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    prior_bounds = {
        "log10_n_initial_samples": (
            args.log10_n_initial_samples_min,
            args.log10_n_initial_samples_max,
        ),
        "observer_speed": (args.observer_speed_min, args.observer_speed_max),
        "dipole_longitude": (args.dipole_longitude_min, args.dipole_longitude_max),
        "dipole_latitude": (args.dipole_latitude_min, args.dipole_latitude_max),
        "temp_beta": (args.temp_beta_min, args.temp_beta_max),
        "elevation_trough": (
            args.elevation_trough_min,
            args.elevation_trough_max,
        ),
        "elevation_amp": (args.elevation_amp_min, args.elevation_amp_max),
        "p_clus": (args.p_clus_min, args.p_clus_max),
        "clus_stop_prob": (args.clus_stop_prob_min, args.clus_stop_prob_max),
        "lambda_clus": (args.lambda_clus_min, args.lambda_clus_max),
    }
    for name, (low, high) in prior_bounds.items():
        if low >= high:
            parser.error(f"--{name}_min must be less than --{name}_max.")


def _parse_modes(raw_modes: list[str] | None, parser: argparse.ArgumentParser) -> list[str]:
    modes: list[str] = []
    for entry in raw_modes or []:
        modes.extend(part.strip().upper() for part in entry.split(",") if part.strip())
    if not modes:
        parser.error("At least one mode must be provided via --mode.")
    return modes


def build_racs_config(
    *,
    catalogue_path: str | Path,
    racs_epoch: str,
    flux_min: float,
    nside: int,
    chunk_size: int,
    use_jax: bool,
    cluster_count_model: str,
    downscale_nside: int | None,
    alpha_mean: float,
    alpha_sigma: float,
    noisemap_data_dir: str,
    noise_map_nside: int,
    flux_error_noise_bins: int,
    flux_error_flux_bins: int,
    flux_error_min_cell_count: int,
    flux_error_noise_bounds_ujy_beam: tuple[float, float],
    flux_error_flux_bounds_mjy: tuple[float, float],
    paf_temperature_data_dir: str | None,
    flux_temperature_min_mjy: float | None = None,
    temperature_model: str = "hot_linear",
    mask_map: np.ndarray | None = None,
    max_cluster_children_per_parent: int = 16,
    temperature_fallback: str = "none",
    paf_reference_temp_c: float | None = None,
    max_reference_fallback_tiles: int | None = None,
) -> RacsConfig:
    if temperature_fallback not in {"none", "open_meteo", "reference"}:
        raise ValueError(f"Unknown temperature fallback: {temperature_fallback}")
    if temperature_fallback == "reference":
        if paf_reference_temp_c is None or not np.isfinite(paf_reference_temp_c):
            raise ValueError(
                "Reference fallback requires a finite paf_reference_temp_c."
            )
        if (
            max_reference_fallback_tiles is None
            or max_reference_fallback_tiles <= 0
        ):
            raise ValueError(
                "Reference fallback requires a positive "
                "max_reference_fallback_tiles."
            )
    elif (
        paf_reference_temp_c is not None
        or max_reference_fallback_tiles is not None
    ):
        raise ValueError(
            "Reference fallback settings may only be used when "
            "temperature_fallback='reference'."
        )
    fallback_config = {}
    if temperature_fallback == "reference":
        fallback_config = {
            "paf_reference_temp_c": paf_reference_temp_c,
            "max_reference_fallback_tiles": max_reference_fallback_tiles,
        }
    return RacsConfig(
        product=racs_epoch,
        catalogue_path=str(Path(catalogue_path).expanduser()),
        flux_min=flux_min,
        nside=nside,
        chunk_size=chunk_size,
        use_float32=False,
        cluster_count_model=cluster_count_model,
        downscale_nside=downscale_nside,
        store_final_samples=not use_jax,
        alpha_mean=alpha_mean,
        alpha_sigma=alpha_sigma,
        noisemap_data_dir=noisemap_data_dir,
        noise_map_nside=noise_map_nside,
        flux_error_noise_bins=flux_error_noise_bins,
        flux_error_flux_bins=flux_error_flux_bins,
        flux_error_min_cell_count=flux_error_min_cell_count,
        flux_error_noise_bounds_ujy_beam=flux_error_noise_bounds_ujy_beam,
        flux_error_flux_bounds_mjy=flux_error_flux_bounds_mjy,
        flux_temperature_min_mjy=flux_temperature_min_mjy,
        temperature_model=temperature_model,
        paf_temperature_data_dir=paf_temperature_data_dir,
        temperature_fallback=temperature_fallback,
        mask_map=mask_map,
        max_cluster_children_per_parent=max_cluster_children_per_parent,
        **fallback_config,
    )


def _summary_ndim(
    summary_features: list[SummaryFeature],
    flux_temperature_n_bins: int,
    flux_temperature_quantiles: tuple[float, ...],
    flux_elevation_n_bins: int,
    flux_elevation_quantiles: tuple[float, ...],
) -> int | None:
    if not summary_features:
        return None
    ndim = 0
    if "log_dispersion" in summary_features:
        ndim += 1
    if "flux_quantiles" in summary_features:
        ndim += _flux_temperature_quantile_ndim(
            flux_temperature_n_bins,
            flux_temperature_quantiles,
        )
    if "flux_elevation_quantiles" in summary_features:
        ndim += _flux_elevation_quantile_ndim(
            flux_elevation_n_bins,
            flux_elevation_quantiles,
        )
    return ndim


def _round_quantile_diagnostic_specs(
    model: Racs | RacsJax,
    summary_features: list[SummaryFeature],
    flux_temperature_n_bins: int,
    flux_temperature_quantiles: tuple[float, ...],
    flux_elevation_n_bins: int,
    flux_elevation_quantiles: tuple[float, ...],
) -> tuple[QuantileSummaryDiagnosticSpec, ...]:
    """Describe quantile blocks in the flattened hybrid data vector."""
    offset = 0
    specs: list[QuantileSummaryDiagnosticSpec] = []
    for summary in summary_features:
        if summary == "log_dispersion":
            offset += 1
        elif summary == "flux_quantiles":
            temperature_edges = _flux_temperature_edges(
                model,
                flux_temperature_n_bins,
            )
            specs.append(
                QuantileSummaryDiagnosticSpec(
                    name="flux_temperature_quantiles",
                    start=offset,
                    bin_edges=tuple(float(value) for value in temperature_edges),
                    quantiles=tuple(flux_temperature_quantiles),
                    x_label="PAF temperature [C]",
                )
            )
            offset += _flux_temperature_quantile_ndim(
                flux_temperature_n_bins,
                flux_temperature_quantiles,
            )
        elif summary == "flux_elevation_quantiles":
            elevation_edges = _flux_elevation_edges(
                model,
                flux_elevation_n_bins,
            )
            specs.append(
                QuantileSummaryDiagnosticSpec(
                    name="flux_elevation_quantiles",
                    start=offset,
                    bin_edges=tuple(float(value) for value in elevation_edges),
                    quantiles=tuple(flux_elevation_quantiles),
                    x_label="Elevation [deg]",
                )
            )
            offset += _flux_elevation_quantile_ndim(
                flux_elevation_n_bins,
                flux_elevation_quantiles,
            )
        else:
            raise ValueError(f"Unknown summary feature: {summary}")
    return tuple(specs)


def build_prior_and_reference_theta(
    simulate_clustering: None | Literal["geometric", "poisson"] = None,
    add_extra_error: bool = False,
    *,
    log10_n_initial_samples_range: tuple[float, float] = (5.6, 6.8),
    observer_speed_range: tuple[float, float] = (0.0, 12.0),
    dipole_longitude_range: tuple[float, float] = (0.0, 360.0),
    dipole_latitude_range: tuple[float, float] = (-90.0, 90.0),
    temp_beta_range: tuple[float, float] = (0.0, 0.05),
    add_elevation_model: bool = False,
    elevation_trough_range: tuple[float, float] = (0.0, 90.0),
    elevation_amp_range: tuple[float, float] = (0.0, 0.2),
    p_clus_range: tuple[float, float] = (0.0, 1.0),
    clus_stop_prob_range: tuple[float, float] = (0.4, 1.0),
    lambda_clus_range: tuple[float, float] = (0.0, 3.0),
    eta_range: tuple[float, float] = (0., 8.)
) -> tuple[DipolePriorNP, dict[str, float]]:
    prior = DipolePriorNP(
        mean_count_range=list(log10_n_initial_samples_range),
        speed_range=list(observer_speed_range),
        longitude_range=list(dipole_longitude_range),
        latitude_range=list(dipole_latitude_range),
    )
    prior.change_kwarg(
        param_short_name="N",
        new_kwarg="log10_n_initial_samples",
    )
    prior.add_prior(
        short_name="beta",
        simulator_kwarg="temp_beta",
        low=temp_beta_range[0],
        high=temp_beta_range[1],
        dist_type="Uniform",
    )
    if add_elevation_model:
        prior.add_prior(
            short_name="hlow",
            simulator_kwarg="elevation_trough",
            low=elevation_trough_range[0],
            high=elevation_trough_range[1],
            dist_type="Uniform",
        )
        prior.add_prior(
            short_name="hamp",
            simulator_kwarg="elevation_amp",
            low=elevation_amp_range[0],
            high=elevation_amp_range[1],
            dist_type="Uniform",
        )
    if simulate_clustering == "geometric":
        prior.add_prior(
            "pclus",
            simulator_kwarg="p_clus",
            low=p_clus_range[0],
            high=p_clus_range[1],
            dist_type="Uniform",
        )
        prior.add_prior(
            "pstop",
            simulator_kwarg="clus_stop_prob",
            low=clus_stop_prob_range[0],
            high=clus_stop_prob_range[1],
            dist_type="Uniform",
        )
    elif simulate_clustering == "poisson":
        prior.add_prior(
            "lclus",
            simulator_kwarg="lambda_clus",
            low=lambda_clus_range[0],
            high=lambda_clus_range[1],
            dist_type="Uniform",
        )
    elif simulate_clustering is not None:
        raise ValueError(f"{simulate_clustering} not recognised.")

    if add_extra_error:
        prior.add_prior(
            'eta',
            simulator_kwarg='fractional_error_eta',
            low=eta_range[0],
            high=eta_range[1],
            dist_type='Uniform'
        )

    theta_0 = {
        "log10_n_initial_samples": 6.65,
        "observer_speed": 1.0,
        "dipole_longitude": 264.021,
        "dipole_latitude": 48.253,
        "temp_beta": 0.02,
        "p_clus": 0.0,
        "clus_stop_prob": 1.0,
        "fractional_error_eta": 0.
    }
    if add_elevation_model:
        theta_0["elevation_trough"] = 0.0
        theta_0["elevation_amp"] = 0.0
    if simulate_clustering == "poisson":
        theta_0["lambda_clus"] = 0.4

    return prior, theta_0


def _generate_dipole_native(
    model: Racs,
    *args,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    original_downscale_nside = model.downscale_nside
    try:
        model.downscale_nside = None
        return model.generate_dipole(*args, **kwargs)
    finally:
        model.downscale_nside = original_downscale_nside


def _generate_dipole_with_flux_summaries_native(
    model: Racs,
    *args,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    original_downscale_nside = model.downscale_nside
    try:
        model.downscale_nside = None
        return model.generate_dipole_with_flux_summaries(*args, **kwargs)
    finally:
        model.downscale_nside = original_downscale_nside


def make_simulator_wrapper(
    model: Racs,
    *,
    native_output: bool = False,
    summary_features: list[SummaryFeature] | None = None,
    flux_temperature_n_bins: int = DEFAULT_FLUX_TEMPERATURE_N_BINS,
    flux_temperature_quantiles: tuple[float, ...] = DEFAULT_FLUX_TEMPERATURE_QUANTILES,
    flux_elevation_n_bins: int = DEFAULT_FLUX_ELEVATION_N_BINS,
    flux_elevation_quantiles: tuple[float, ...] = DEFAULT_FLUX_ELEVATION_QUANTILES,
):
    summary_features = list(summary_features or [])

    def simulator_wrapper(
        rng_key: NPKey | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        if native_output:
            return _generate_dipole_native(model, rng_key=rng_key, **kwargs)

        if not summary_features:
            return model.generate_dipole(rng_key=rng_key, **kwargs)

        has_temperature_summary = "flux_quantiles" in summary_features
        has_elevation_summary = "flux_elevation_quantiles" in summary_features
        summary_kwargs = {}
        if has_temperature_summary:
            summary_kwargs.update(
                temperature_edges=_flux_temperature_edges(
                    model,
                    flux_temperature_n_bins,
                ),
                temperature_quantiles=flux_temperature_quantiles,
            )
        if has_elevation_summary:
            summary_kwargs.update(
                elevation_edges=_flux_elevation_edges(
                    model,
                    flux_elevation_n_bins,
                ),
                elevation_quantiles=flux_elevation_quantiles,
            )

        if summary_kwargs:
            native_map, native_mask, source_summaries = (
                _generate_dipole_with_flux_summaries_native(
                    model,
                    rng_key=rng_key,
                    **summary_kwargs,
                    **kwargs,
                )
            )
        else:
            native_map, native_mask = _generate_dipole_native(
                model,
                rng_key=rng_key,
                **kwargs,
            )
            source_summaries = {}

        summary_parts: list[np.ndarray] = []
        for summary in summary_features:
            if summary == "log_dispersion":
                summary_parts.append(
                    _native_count_log_dispersion_feature(native_map, native_mask)
                )
            elif summary == "flux_quantiles":
                summary_parts.append(source_summaries["temperature"])
            elif summary == "flux_elevation_quantiles":
                summary_parts.append(source_summaries["elevation"])
            else:
                raise ValueError(f"Unknown summary feature: {summary}")
        summary_values = np.concatenate(summary_parts).astype(np.float32, copy=False)
        return build_hybrid_sample_from_native(
            native_map,
            native_mask,
            downscale_nside=model.downscale_nside,
            summary_features=summary_features,
            summary_values=summary_values,
        )

    return simulator_wrapper


def make_model_sim_wrapper(
    simulator_wrapper,
    n_workers: int | None,
):
    def model_sim_wrapper(
        npkey: NPKey,
        params: dict[str, np.ndarray],
        noise: bool = True,
        ui: MultiRoundInfererUI | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return batch_simulate(
            params,
            simulator_wrapper,
            n_workers=n_workers,
            ui=ui,
            rng_key=npkey,
        )

    return model_sim_wrapper


def _jax_key_from_npkey(key: NPKey) -> jax.Array:
    if isinstance(key, NPKey):
        key_data = key._ss.generate_state(2, dtype=np.uint32)
    else:
        key_data = np.asarray(key, dtype=np.uint32).reshape(2)
    return jax.device_put(key_data)


def _batch_generate_dipole_native(
    model: RacsJax,
    theta: dict[str, np.ndarray],
    key: jax.Array,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    original_downscale_nside = model.downscale_nside
    try:
        model.downscale_nside = None
        return model.batch_generate_dipole(
            theta,
            key,
            batch_size=batch_size,
            show_progress=True,
        )
    finally:
        model.downscale_nside = original_downscale_nside


def make_jax_model_sim_wrapper(
    model: RacsJax,
    batch_size: int,
    *,
    summary_features: list[SummaryFeature] | None = None,
    flux_temperature_n_bins: int = DEFAULT_FLUX_TEMPERATURE_N_BINS,
    flux_temperature_quantiles: tuple[float, ...] = DEFAULT_FLUX_TEMPERATURE_QUANTILES,
    flux_temperature_jax_flux_bins: int = DEFAULT_FLUX_TEMPERATURE_JAX_FLUX_BINS,
    flux_elevation_n_bins: int = DEFAULT_FLUX_ELEVATION_N_BINS,
    flux_elevation_quantiles: tuple[float, ...] = DEFAULT_FLUX_ELEVATION_QUANTILES,
):
    summary_features = list(summary_features or [])

    def model_sim_wrapper(
        npkey: NPKey,
        params: dict[str, np.ndarray],
        noise: bool = True,
        ui: MultiRoundInfererUI | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        jax_key = _jax_key_from_npkey(npkey)
        theta = {key: np.asarray(value) for key, value in params.items()}

        if not summary_features:
            return model.batch_generate_dipole(
                theta,
                jax_key,
                batch_size=batch_size,
                show_progress=True,
            )

        has_temperature_summary = "flux_quantiles" in summary_features
        has_elevation_summary = "flux_elevation_quantiles" in summary_features
        if has_temperature_summary or has_elevation_summary:
            summary_kwargs = {}
            if has_temperature_summary:
                summary_kwargs.update(
                    temperature_edges=_flux_temperature_edges(
                        model,
                        flux_temperature_n_bins,
                    ),
                    temperature_quantiles=flux_temperature_quantiles,
                )
            if has_elevation_summary:
                summary_kwargs.update(
                    elevation_edges=_flux_elevation_edges(
                        model,
                        flux_elevation_n_bins,
                    ),
                    elevation_quantiles=flux_elevation_quantiles,
                )
            original_downscale_nside = model.downscale_nside
            try:
                model.downscale_nside = None
                native_maps, native_masks, flux_summaries = (
                    model.batch_generate_dipole_with_flux_summaries(
                        theta,
                        jax_key,
                        batch_size=batch_size,
                        n_flux_bins=flux_temperature_jax_flux_bins,
                        show_progress=True,
                        **summary_kwargs,
                    )
                )
            finally:
                model.downscale_nside = original_downscale_nside
        else:
            native_maps, native_masks = _batch_generate_dipole_native(
                model,
                theta,
                jax_key,
                batch_size=batch_size,
            )
            flux_summaries = {}
        outputs = []
        for index, (native_map, native_mask) in enumerate(zip(native_maps, native_masks)):
            summary_parts: list[np.ndarray] = []
            for summary in summary_features:
                if summary == "log_dispersion":
                    summary_parts.append(
                        _native_count_log_dispersion_feature(native_map, native_mask)
                    )
                elif summary == "flux_quantiles":
                    if "temperature" not in flux_summaries:
                        raise ValueError("JAX flux quantile summary was not computed.")
                    summary_parts.append(
                        np.asarray(flux_summaries["temperature"][index], dtype=np.float32)
                    )
                elif summary == "flux_elevation_quantiles":
                    if "elevation" not in flux_summaries:
                        raise ValueError(
                            "JAX elevation flux quantile summary was not computed."
                        )
                    summary_parts.append(
                        np.asarray(flux_summaries["elevation"][index], dtype=np.float32)
                    )
                else:
                    raise ValueError(f"Unknown summary feature: {summary}")
            summary_values = (
                np.concatenate(summary_parts, axis=0).astype(np.float32, copy=False)
                if summary_parts else None
            )
            outputs.append(
                build_hybrid_sample_from_native(
                    native_map,
                    native_mask,
                    downscale_nside=model.downscale_nside,
                    summary_features=summary_features,
                    summary_values=summary_values,
                )
            )
        x = np.stack([output[0] for output in outputs], axis=0)
        mask = np.stack([output[1] for output in outputs], axis=0)
        return x, mask

    return model_sim_wrapper


def build_scenario(
    inference_config: Mapping[str, Any],
    effective_nside: int,
    prior: DipolePriorNP,
    theta_0: dict[str, float],
    out_dir: str,
    ssnle_seed: int,
    n_rounds: int,
    n_simulations: int,
    map_ndim: int | None = None,
    summary_ndim: int | None = None,
    summary_features: list[SummaryFeature] | None = None,
) -> Scenario:
    prior_jax = prior.to_jax()
    summary_features = list(summary_features or [])

    mode = inference_config["mode"]
    scenario_name = inference_config["scenario"]
    scenario_factories = {
        "anynside_nle": Scenario.anynside_nle,
        "anynside_npe": Scenario.anynside_npe,
    }
    scenario_modes = {
        "anynside_nle": "NLE",
        "anynside_npe": "NPE",
    }
    try:
        scenario_factory = scenario_factories[scenario_name]
    except KeyError as error:
        raise ValueError(f"Unknown inference scenario: {scenario_name}") from error
    if scenario_modes[scenario_name] != mode:
        raise ValueError(
            f"Inference mode {mode} is incompatible with scenario {scenario_name}."
        )

    data_spec_values = dict(inference_config["data_transform"])
    embedding_values = data_spec_values.pop("embedding_config", None)
    if embedding_values is not None:
        data_spec_values["embedding_config"] = EmbeddingNetConfig(
            **embedding_values
        )
    data_spec = DataTransformSpec(**data_spec_values)
    theta_spec = ThetaTransformSpec(**inference_config["theta_transform"])

    multiround_overrides = dict(inference_config["multiround"])
    multiround_overrides.update(
        {
            "prng_integer_seed": ssnle_seed,
            "plot_save_dir": out_dir,
            "simulation_budget": n_simulations,
            "n_rounds": n_rounds,
        }
    )
    if mode == "NLE":
        multiround_overrides.update(
            {
                "map_ndim": map_ndim,
                "summary_ndim": summary_ndim,
                "native_count_summary": (
                    "log_dispersion"
                    if "log_dispersion" in summary_features
                    else None
                ),
            }
        )

    return scenario_factory(
        nside=effective_nside,
        theta_prior=prior_jax,
        reference_theta=theta_0,
        training_overrides=dict(inference_config["training"]),
        multiround_overrides=multiround_overrides,
        flow_overrides=dict(inference_config["flow"]),
        data_spec=data_spec,
        theta_spec=theta_spec,
    )


def main() -> None:
    args, parser = construct_argparser()
    inference_config_path = args.inference_config.expanduser().resolve(strict=True)
    inference_config = load_inference_config(inference_config_path)
    configured_mode = str(inference_config["mode"]).upper()
    if args.mode is not None:
        requested_modes = _parse_modes(args.mode, parser)
        if requested_modes != [configured_mode]:
            parser.error(
                "--mode must match the single mode declared by "
                f"--inference_config ({configured_mode})."
            )
    args.mode = [configured_mode]
    modes = validate_args(args, parser)
    summary_features = list(args.summary_features)

    catalogue_path = args.catalogue_path.expanduser().resolve(strict=True)
    reference_observation_path = (
        args.reference_observation.expanduser().resolve(strict=True)
    )
    observation_config_path = (
        args.observation_config.expanduser().resolve(strict=True)
    )
    observation_config = load_observation_config(observation_config_path)
    x0, mask0 = load_reference_observation(reference_observation_path)
    mask = build_mask_from_observation_config(observation_config)
    config = build_racs_config(
        catalogue_path=catalogue_path,
        racs_epoch=args.racs_epoch,
        flux_min=args.flux_min,
        nside=args.nside,
        chunk_size=args.chunk_size,
        use_jax=args.use_jax,
        # catsim requires a count-model implementation even when clustering is
        # disabled. Poisson with its default lambda_clus=0 is the exact no-op.
        cluster_count_model=args.simulate_clustering or "poisson",
        downscale_nside=args.downscale_nside,
        alpha_mean=args.alpha_mean,
        alpha_sigma=args.alpha_sigma,
        noisemap_data_dir=args.noisemap_data_dir,
        noise_map_nside=args.noise_map_nside,
        flux_error_noise_bins=args.flux_error_noise_bins,
        flux_error_flux_bins=args.flux_error_flux_bins,
        flux_error_min_cell_count=args.flux_error_min_cell_count,
        flux_error_noise_bounds_ujy_beam=tuple(
            args.flux_error_noise_bounds_ujy_beam
        ),
        flux_error_flux_bounds_mjy=tuple(args.flux_error_flux_bounds_mjy),
        flux_temperature_min_mjy=args.flux_temperature_min_mjy,
        temperature_model=args.temperature_model,
        temperature_fallback=args.temperature_fallback,
        paf_reference_temp_c=args.paf_reference_temp_c,
        max_reference_fallback_tiles=args.max_reference_fallback_tiles,
        mask_map=mask,
        max_cluster_children_per_parent=args.max_children,
        paf_temperature_data_dir=args.paf_temperature_data_dir,
    )

    model = RacsJax(config) if args.use_jax else Racs(config)
    model.initialise_data()

    if summary_features:
        effective_nside = args.downscale_nside or args.nside
        map_ndim = hp.nside2npix(effective_nside)
        summary_ndim = _summary_ndim(
            summary_features,
            args.flux_temperature_n_bins,
            args.flux_temperature_quantiles,
            args.flux_elevation_n_bins,
            args.flux_elevation_quantiles,
        )
        expected_data_ndim = map_ndim + summary_ndim
        if x0.size != expected_data_ndim:
            raise ValueError(
                "Prepared reference observation is incompatible with the configured "
                "map and summary dimensions: "
                f"expected {expected_data_ndim} values, found {x0.size} in "
                f"{reference_observation_path}."
            )
    else:
        effective_nside = hp.npix2nside(x0.size)
        map_ndim = None
        summary_ndim = None

    round_quantile_specs = (
        _round_quantile_diagnostic_specs(
            model,
            summary_features,
            args.flux_temperature_n_bins,
            args.flux_temperature_quantiles,
            args.flux_elevation_n_bins,
            args.flux_elevation_quantiles,
        )
        if map_ndim is not None and summary_features
        else ()
    )

    observed_map = x0[:map_ndim] if map_ndim is not None else x0
    observed_count = float(np.nansum(observed_map))
    if observed_count <= 0:
        product = _model_product(model)
        raise ValueError(
            f"Observed {product.label} map has zero total counts after masking/cuts."
        )

    prior, theta_0 = build_prior_and_reference_theta(
        simulate_clustering=args.simulate_clustering,
        add_extra_error=args.add_extra_error,
        log10_n_initial_samples_range=(
            args.log10_n_initial_samples_min,
            args.log10_n_initial_samples_max,
        ),
        observer_speed_range=(args.observer_speed_min, args.observer_speed_max),
        dipole_longitude_range=(
            args.dipole_longitude_min,
            args.dipole_longitude_max,
        ),
        dipole_latitude_range=(args.dipole_latitude_min, args.dipole_latitude_max),
        temp_beta_range=(args.temp_beta_min, args.temp_beta_max),
        add_elevation_model=args.add_elevation_model,
        elevation_trough_range=(
            args.elevation_trough_min,
            args.elevation_trough_max,
        ),
        elevation_amp_range=(args.elevation_amp_min, args.elevation_amp_max),
        p_clus_range=(args.p_clus_min, args.p_clus_max),
        clus_stop_prob_range=(args.clus_stop_prob_min, args.clus_stop_prob_max),
        lambda_clus_range=(args.lambda_clus_min, args.lambda_clus_max),
        eta_range=(args.eta_min, args.eta_max)
    )

    if args.use_jax:
        model_sim_wrapper = make_jax_model_sim_wrapper(
            model,
            batch_size=args.jax_batch_size,
            summary_features=summary_features,
            flux_temperature_n_bins=args.flux_temperature_n_bins,
            flux_temperature_quantiles=args.flux_temperature_quantiles,
            flux_elevation_n_bins=args.flux_elevation_n_bins,
            flux_elevation_quantiles=args.flux_elevation_quantiles,
        )
    else:
        simulator_wrapper = make_simulator_wrapper(
            model,
            summary_features=summary_features,
            flux_temperature_n_bins=args.flux_temperature_n_bins,
            flux_temperature_quantiles=args.flux_temperature_quantiles,
            flux_elevation_n_bins=args.flux_elevation_n_bins,
            flux_elevation_quantiles=args.flux_elevation_quantiles,
        )
        model_sim_wrapper = make_model_sim_wrapper(
            simulator_wrapper=simulator_wrapper,
            n_workers=args.n_workers,
        )

    for mode in modes:
        mode_out_dir = (
            args.out_dir
            if len(modes) == 1
            else str(Path(args.out_dir) / mode)
        )
        scenario = build_scenario(
            inference_config=inference_config,
            effective_nside=effective_nside,
            prior=prior,
            theta_0=theta_0,
            out_dir=mode_out_dir,
            ssnle_seed=args.ssnle_seed,
            n_rounds=args.n_rounds,
            n_simulations=args.n_simulations,
            map_ndim=map_ndim,
            summary_ndim=summary_ndim,
            summary_features=summary_features,
        )
        inferer = MultiRoundInferer(
            mode,
            prior,
            model_sim_wrapper,
            (x0, mask0),
            multi_round_config=scenario.multiround,
            transform_config=scenario.transforms,
            nflow_config=scenario.flow,
            train_config=scenario.training,
            use_ui=not args.no_ui,
            model_config=config,
            round_simulation_diagnostic=(
                make_round_quantile_diagnostic(round_quantile_specs)
                if mode == "NLE" and round_quantile_specs
                else None
            ),
        )
        _write_run_command(inferer.mr_config.plot_save_dir)
        inferer.run()


if __name__ == "__main__":
    main()
