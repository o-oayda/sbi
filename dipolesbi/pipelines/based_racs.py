import argparse
from typing import Literal

from catsim import RACS_PRODUCTS, Racs, RacsConfig
from catsim.racs_jax import RacsJax
from catsim.utils.healsphere import downgrade_ignore_nan
from dipoleutils.utils.data_loader import DataLoader
from dipoleutils.utils.mask import Masker
from dipoleutils.utils.samples import CatalogueToMap
import healpy as hp
import jax
import matplotlib.pyplot as plt
import numpy as np

from dipolesbi.pipelines.summary_stats import (
    _flux_elevation_edges,
    _flux_elevation_histogram_quantile_features,
    _flux_elevation_quantile_features,
    _flux_elevation_quantile_ndim,
    _flux_temperature_edges,
    _flux_temperature_histogram_quantile_features,
    _flux_temperature_quantile_features,
    _flux_temperature_quantile_ndim,
    _native_count_log_dispersion_feature,
    _real_catalogue_flux_elevation_samples,
    _real_catalogue_flux_temperature_samples,
)
from dipolesbi.tools.configs import DataTransformSpec, Scenario
from dipolesbi.tools.multiround_inferer import MultiRoundInferer
from dipolesbi.tools.np_rngkey import NPKey
from dipolesbi.tools.priors_np import DipolePriorNP
from dipolesbi.tools.ui import MultiRoundInfererUI
from dipolesbi.tools.utils import batch_simulate

SummaryFeature = Literal[
    "log_dispersion",
    "flux_quantiles",
    "flux_elevation_quantiles",
]
DEFAULT_FLUX_TEMPERATURE_N_BINS = 10
DEFAULT_FLUX_TEMPERATURE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
DEFAULT_FLUX_ELEVATION_N_BINS = 10
DEFAULT_FLUX_ELEVATION_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
DEFAULT_FLUX_TEMPERATURE_JAX_FLUX_BINS = 128
DEFAULT_PAF_TEMPERATURE_DATA_DIR = "/home/oliver/Documents/dipole-utils/data/paf_temps"


def construct_argparser() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        nargs="+",
        help='One or more modes to run, separated by spaces (e.g. "--mode NLE NPE").',
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
        help="Directory to save outputs into.",
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
        "--racs_epoch",
        choices=sorted(RACS_PRODUCTS),
        default="low3",
        help="RACS product/epoch to simulate.",
    )
    parser.add_argument(
        "--openmeteo_fallback",
        action="store_true",
        help="Use Open-Meteo ambient temperatures when PAF temperatures are unavailable.",
    )
    parser.add_argument(
        "--paf_temperature_data_dir",
        type=str,
        default=DEFAULT_PAF_TEMPERATURE_DATA_DIR,
        help="Directory containing PAF temperature data.",
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
        "--fractional_error_flux_min_mjy",
        type=float,
        default=10.0,
        help="Minimum flux used when building the empirical fractional-error lookup.",
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
    parser.add_argument("--max_children", type=int, default=16)
    return parser.parse_args(), parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    if args.chunk_size is None:
        args.chunk_size = 140_000 if args.use_jax else 2_500_000
    if args.use_jax and args.n_workers is not None:
        parser.error("--n_workers is only used by the NumPy simulator.")
    if args.use_jax and args.jax_batch_size <= 0:
        parser.error("--jax_batch_size must be positive.")
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
    _validate_prior_bounds(args, parser)

    modes = _parse_modes(args.mode, parser)
    if args.summary_features and any(mode != "NLE" for mode in modes):
        parser.error("Summary statistics are only supported for the NLE at the moment.")

    if not args.simulate_clustering:
        args.simulate_clustering = "geometric"
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
    racs_epoch: str,
    flux_min: float,
    nside: int,
    chunk_size: int,
    use_jax: bool,
    cluster_count_model: str,
    downscale_nside: int | None,
    alpha_mean: float,
    alpha_sigma: float,
    fractional_error_flux_min_mjy: float,
    mask_map: np.ndarray | None = None,
    max_cluster_children_per_parent: int = 16,
    paf_temperature_data_dir: str = DEFAULT_PAF_TEMPERATURE_DATA_DIR,
    openmeteo_fallback: bool = False,
) -> RacsConfig:
    return RacsConfig(
        product=racs_epoch,
        flux_min=flux_min,
        nside=nside,
        chunk_size=chunk_size,
        use_float32=False,
        cluster_count_model=cluster_count_model,
        downscale_nside=downscale_nside,
        store_final_samples=not use_jax,
        alpha_mean=alpha_mean,
        alpha_sigma=alpha_sigma,
        fractional_error_flux_min_mjy=fractional_error_flux_min_mjy,
        paf_temperature_data_dir=paf_temperature_data_dir,
        temperature_fallback="open_meteo" if openmeteo_fallback else "none",
        mask_map=mask_map,
        max_cluster_children_per_parent=max_cluster_children_per_parent,
    )


def build_mask(nside: int) -> np.ndarray:
    masker = Masker(np.ones(hp.nside2npix(nside)), coordinate_system="equatorial")
    masker.mask_galactic_plane(5)
    masker.mask_a_team_sources(radius_deg=2)
    masker.mask_equatorial_poles(north_radius=42)
    masker.mask_a_team_sources(radius_deg=3, source_names=["Cygnus A"])
    masker.mask_a_team_sources(radius_deg=13, source_names=["LMC"])
    masker.mask_a_team_sources(radius_deg=8, source_names=["SMC"])
    return hp.reorder(masker.get_mask_map(), r2n=True)


def _model_product(model: Racs | RacsJax):
    product = getattr(model, "product", None)
    if product is not None:
        return product
    return model.cfg.product


def _append_summary_features(
    data: np.ndarray,
    mask: np.ndarray,
    summary_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    summary_features = np.asarray(summary_features, dtype=np.float32)
    if summary_features.ndim != 1:
        raise ValueError("summary_features must be one-dimensional.")
    summary_mask = np.ones(summary_features.shape, dtype=np.bool_)
    hybrid = np.concatenate([np.asarray(data, dtype=np.float32), summary_features])
    hybrid_mask = np.concatenate([np.asarray(mask, dtype=np.bool_), summary_mask])
    return hybrid.astype(np.float32, copy=False), hybrid_mask.astype(np.bool_, copy=False)


def _prepare_output_map(
    native_map: np.ndarray,
    native_mask: np.ndarray,
    downscale_nside: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if downscale_nside is None:
        output_map = np.asarray(native_map, dtype=np.float32).copy()
        output_mask = np.asarray(native_mask, dtype=np.bool_)
    else:
        output_map, output_mask = downgrade_ignore_nan(
            native_map,
            native_mask,
            downscale_nside,
        )
        output_map = output_map.astype(np.float32, copy=False)
        output_mask = output_mask.astype(np.bool_, copy=False)
        output_map = output_map.copy()
    output_map[~output_mask] = np.nan
    return output_map, output_mask


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


def _flux_temperature_summary(
    model: Racs,
    flux: np.ndarray,
    temperature: np.ndarray,
    flux_temperature_n_bins: int,
    flux_temperature_quantiles: tuple[float, ...],
) -> np.ndarray:
    return _flux_temperature_quantile_features(
        flux,
        temperature,
        temp_edges=_flux_temperature_edges(model, flux_temperature_n_bins),
        quantiles=flux_temperature_quantiles,
    )


def _flux_elevation_summary(
    model: Racs,
    flux: np.ndarray,
    elevation: np.ndarray,
    flux_elevation_n_bins: int,
    flux_elevation_quantiles: tuple[float, ...],
) -> np.ndarray:
    return _flux_elevation_quantile_features(
        flux,
        elevation,
        elevation_edges=_flux_elevation_edges(model, flux_elevation_n_bins),
        quantiles=flux_elevation_quantiles,
    )


def _build_summary_features(
    native_map: np.ndarray,
    native_mask: np.ndarray,
    summary_features: list[SummaryFeature],
    *,
    model: Racs | None = None,
    flux: np.ndarray | None = None,
    temperature: np.ndarray | None = None,
    elevation_flux: np.ndarray | None = None,
    elevation: np.ndarray | None = None,
    flux_temperature_n_bins: int = DEFAULT_FLUX_TEMPERATURE_N_BINS,
    flux_temperature_quantiles: tuple[float, ...] = DEFAULT_FLUX_TEMPERATURE_QUANTILES,
    flux_elevation_n_bins: int = DEFAULT_FLUX_ELEVATION_N_BINS,
    flux_elevation_quantiles: tuple[float, ...] = DEFAULT_FLUX_ELEVATION_QUANTILES,
) -> np.ndarray:
    stats: list[np.ndarray] = []
    for summary in summary_features:
        if summary == "log_dispersion":
            stats.append(_native_count_log_dispersion_feature(native_map, native_mask))
        elif summary == "flux_quantiles":
            if model is None or flux is None or temperature is None:
                raise ValueError("Flux quantiles require model, flux, and temperature.")
            stats.append(
                _flux_temperature_summary(
                    model,
                    flux,
                    temperature,
                    flux_temperature_n_bins,
                    flux_temperature_quantiles,
                )
            )
        elif summary == "flux_elevation_quantiles":
            if model is None or elevation_flux is None or elevation is None:
                raise ValueError(
                    "Elevation flux quantiles require model, flux, and elevation."
                )
            stats.append(
                _flux_elevation_summary(
                    model,
                    elevation_flux,
                    elevation,
                    flux_elevation_n_bins,
                    flux_elevation_quantiles,
                )
            )
        else:
            raise ValueError(f"Unknown summary feature: {summary}")
    if not stats:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(stats, axis=0).astype(np.float32, copy=False)


def build_hybrid_sample_from_native(
    native_map: np.ndarray,
    native_mask: np.ndarray,
    *,
    downscale_nside: int | None,
    summary_features: list[SummaryFeature],
    summary_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    output_map, output_mask = _prepare_output_map(
        native_map,
        native_mask,
        downscale_nside,
    )
    if not summary_features and summary_values is None:
        return output_map, output_mask
    if summary_values is None:
        summary_values = _build_summary_features(
            native_map,
            native_mask,
            summary_features,
        )
    return _append_summary_features(output_map, output_mask, summary_values)


def build_real_sample(
    model: Racs | RacsJax,
    flux_min: float,
    summary_features: list[SummaryFeature] | None = None,
    *,
    flux_temperature_n_bins: int = DEFAULT_FLUX_TEMPERATURE_N_BINS,
    flux_temperature_quantiles: tuple[float, ...] = DEFAULT_FLUX_TEMPERATURE_QUANTILES,
    flux_elevation_n_bins: int = DEFAULT_FLUX_ELEVATION_N_BINS,
    flux_elevation_quantiles: tuple[float, ...] = DEFAULT_FLUX_ELEVATION_QUANTILES,
    save_map_plot: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    summary_features = list(summary_features or [])

    product = _model_product(model)
    cat = DataLoader(*product.data_loader_args).load()
    c2map = CatalogueToMap(cat)
    c2map.make_cut(product.columns.total_flux, minimum=flux_min, maximum=None)
    if product.columns.source_name in cat.colnames:
        c2map.crossmatch_local_sources(
            "equatorial",
            radius=5,
            source_name_A_column=product.columns.source_name,
        )
    density_map = c2map.make_density_map(
        coordinate_system="equatorial",
        nside=model.nside,
        nest=True,
    ).astype("float32")

    native_mask = model.mask_map.astype(np.bool_, copy=False)
    native_map = density_map.copy()
    native_map[~native_mask] = np.nan

    if save_map_plot:
        hp.projview(native_map, nest=True)
        plt.savefig(f"racs_{product.key}.png")
        plt.close()

    flux = temperature = elevation_flux = elevation = None
    summary_values: np.ndarray | None = None
    if "flux_quantiles" in summary_features:
        flux, temperature = _real_catalogue_flux_temperature_samples(model, c2map)
    if "flux_elevation_quantiles" in summary_features:
        elevation_flux, elevation = _real_catalogue_flux_elevation_samples(model, c2map)
    if isinstance(model, RacsJax) and (
        "flux_quantiles" in summary_features
        or "flux_elevation_quantiles" in summary_features
    ):
        flux_max = model.flux_summary_flux_max_mjy
        if flux_max is None:
            raise ValueError("JAX flux summary requires flux_max.")
        stats: list[np.ndarray] = []
        for summary in summary_features:
            if summary == "log_dispersion":
                stats.append(_native_count_log_dispersion_feature(native_map, native_mask))
            elif summary == "flux_quantiles":
                stats.append(
                    _flux_temperature_histogram_quantile_features(
                        flux,
                        temperature,
                        temp_edges=_flux_temperature_edges(
                            model,
                            flux_temperature_n_bins,
                        ),
                        quantiles=flux_temperature_quantiles,
                        flux_min_mjy=flux_min,
                        flux_max_mjy=flux_max,
                        n_flux_bins=DEFAULT_FLUX_TEMPERATURE_JAX_FLUX_BINS,
                    )
                )
            elif summary == "flux_elevation_quantiles":
                stats.append(
                    _flux_elevation_histogram_quantile_features(
                        elevation_flux,
                        elevation,
                        elevation_edges=_flux_elevation_edges(
                            model,
                            flux_elevation_n_bins,
                        ),
                        quantiles=flux_elevation_quantiles,
                        flux_min_mjy=flux_min,
                        flux_max_mjy=flux_max,
                        n_flux_bins=DEFAULT_FLUX_TEMPERATURE_JAX_FLUX_BINS,
                    )
                )
            else:
                raise ValueError(f"Unknown summary feature: {summary}")
        summary_values = np.concatenate(stats, axis=0).astype(np.float32, copy=False)
    if summary_values is None:
        summary_values = _build_summary_features(
            native_map,
            native_mask,
            summary_features,
            model=model if isinstance(model, Racs) else None,
            flux=flux,
            temperature=temperature,
            elevation_flux=elevation_flux,
            elevation=elevation,
            flux_temperature_n_bins=flux_temperature_n_bins,
            flux_temperature_quantiles=flux_temperature_quantiles,
            flux_elevation_n_bins=flux_elevation_n_bins,
            flux_elevation_quantiles=flux_elevation_quantiles,
        )
    return build_hybrid_sample_from_native(
        native_map,
        native_mask,
        downscale_nside=model.downscale_nside,
        summary_features=summary_features,
        summary_values=summary_values,
    )


def build_prior_and_reference_theta(
    simulate_clustering: None | Literal["geometric", "poisson"] = None,
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

    theta_0 = {
        "log10_n_initial_samples": 6.65,
        "observer_speed": 1.0,
        "dipole_longitude": 264.021,
        "dipole_latitude": 48.253,
        "temp_beta": 0.02,
        "p_clus": 0.0,
        "clus_stop_prob": 1.0,
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

        native_map, native_mask = _generate_dipole_native(
            model,
            rng_key=rng_key,
            **kwargs,
        )
        flux = temperature = elevation_flux = elevation = None
        if "flux_quantiles" in summary_features:
            if getattr(model, "final_observed_flux_samples", None) is None:
                raise ValueError(
                    "Flux-temperature summary requires stored final flux samples."
                )
            if getattr(model, "final_temperature_samples", None) is None:
                raise ValueError(
                    "Flux-temperature summary requires stored final temperature samples."
                )
            flux = model.final_observed_flux_samples
            temperature = model.final_temperature_samples
        if "flux_elevation_quantiles" in summary_features:
            if getattr(model, "final_observed_flux_samples", None) is None:
                raise ValueError(
                    "Flux-elevation summary requires stored final flux samples."
                )
            if getattr(model, "final_elevation_samples", None) is None:
                raise ValueError(
                    "Flux-elevation summary requires stored final elevation samples."
                )
            elevation_flux = model.final_observed_flux_samples
            elevation = model.final_elevation_samples

        summary_values = _build_summary_features(
            native_map,
            native_mask,
            summary_features,
            model=model,
            flux=flux,
            temperature=temperature,
            elevation_flux=elevation_flux,
            elevation=elevation,
            flux_temperature_n_bins=flux_temperature_n_bins,
            flux_temperature_quantiles=flux_temperature_quantiles,
            flux_elevation_n_bins=flux_elevation_n_bins,
            flux_elevation_quantiles=flux_elevation_quantiles,
        )
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
    mode: str,
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

    if mode == "NPE":
        return Scenario.anynside_npe(
            nside=effective_nside,
            theta_prior=prior_jax,
            reference_theta=theta_0,
            theta_spec_overrides={"embed_transform_in_flow": True},
            multiround_overrides={
                "prng_integer_seed": ssnle_seed,
                "plot_save_dir": out_dir,
                "n_rounds": n_rounds,
                "simulation_budget": n_simulations,
                "likelihood_chunk_size_gb": 0.5,
                "n_likelihood_samples": 10_000,
            },
            training_overrides={"learning_rate": 0.001},
        )

    if mode == "NLE":
        return Scenario.anynside_nle(
            nside=effective_nside,
            theta_prior=prior_jax,
            training_overrides={
                "learning_rate": 1e-4,
                "min_lr_ratio": 1.0,
            },
            reference_theta=theta_0,
            multiround_overrides={
                "prng_integer_seed": ssnle_seed,
                "plot_save_dir": out_dir,
                "simulation_budget": n_simulations,
                "n_rounds": n_rounds,
                "likelihood_chunk_size_gb": 0.5,
                "n_likelihood_samples": 10_000,
                "map_ndim": map_ndim,
                "summary_ndim": summary_ndim,
                "native_count_summary": (
                    "log_dispersion" if "log_dispersion" in summary_features else None
                ),
            },
            flow_overrides={
                "decoder_n_neurons": 128,
                "decoder_n_layers": 4,
                "architecture": 11 * ["MAF"],
                "data_reduction_factor": 0.5,
            },
            data_spec=DataTransformSpec.zscore(method="batchwise"),
        )

    raise KeyError(f"Mode {mode} not recognised.")


def main() -> None:
    args, parser = construct_argparser()
    modes = validate_args(args, parser)
    summary_features = list(args.summary_features)

    mask = build_mask(args.nside)
    config = build_racs_config(
        racs_epoch=args.racs_epoch,
        flux_min=args.flux_min,
        nside=args.nside,
        chunk_size=args.chunk_size,
        use_jax=args.use_jax,
        cluster_count_model=args.simulate_clustering,
        downscale_nside=args.downscale_nside,
        alpha_mean=args.alpha_mean,
        alpha_sigma=args.alpha_sigma,
        fractional_error_flux_min_mjy=args.fractional_error_flux_min_mjy,
        openmeteo_fallback=args.openmeteo_fallback,
        mask_map=mask,
        max_cluster_children_per_parent=args.max_children,
        paf_temperature_data_dir=args.paf_temperature_data_dir,
    )

    model = RacsJax(config) if args.use_jax else Racs(config)
    model.initialise_data()

    x0, mask0 = build_real_sample(
        model,
        args.flux_min,
        summary_features,
        flux_temperature_n_bins=args.flux_temperature_n_bins,
        flux_temperature_quantiles=args.flux_temperature_quantiles,
        flux_elevation_n_bins=args.flux_elevation_n_bins,
        flux_elevation_quantiles=args.flux_elevation_quantiles,
    )

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
    else:
        effective_nside = hp.npix2nside(x0.size)
        map_ndim = None
        summary_ndim = None

    observed_map = x0[:map_ndim] if map_ndim is not None else x0
    observed_count = float(np.nansum(observed_map))
    if observed_count <= 0:
        product = _model_product(model)
        raise ValueError(
            f"Observed {product.label} map has zero total counts after masking/cuts."
        )

    prior, theta_0 = build_prior_and_reference_theta(
        simulate_clustering=args.simulate_clustering,
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
        scenario = build_scenario(
            mode=mode,
            effective_nside=effective_nside,
            prior=prior,
            theta_0=theta_0,
            out_dir=args.out_dir,
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
        )
        inferer.run()


if __name__ == "__main__":
    main()
