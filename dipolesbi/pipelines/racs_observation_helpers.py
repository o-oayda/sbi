from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from astropy.table import Table
from catsim import Racs, RacsProductSpec
from catsim.racs_jax import RacsJax
from catsim.utils.healsphere import downgrade_ignore_nan
from dipoleutils.utils.mask import Masker
from dipoleutils.utils.samples import CatalogueToMap
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from dipolesbi.pipelines.summary_stats import (
    _flux_elevation_edges,
    _flux_elevation_histogram_quantile_features,
    _flux_elevation_quantile_features,
    _flux_temperature_edges,
    _flux_temperature_histogram_quantile_features,
    _flux_temperature_quantile_features,
    _native_count_log_dispersion_feature,
    _real_catalogue_flux_elevation_samples,
    _real_catalogue_flux_temperature_samples,
)
from dipolesbi.pipelines.experiment_config import (
    ObservationConfigError,
    resolve_observation_config,
)
from dipolesbi.lib.yaml_to_mask import yaml_to_mask


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

DEFAULT_SOURCE_RADII_DEG: dict[str, float] = {
    "Cygnus A": 3,
    "LMC": 13,
    "SMC": 8,
}


def load_observation_config(config_path: str | Path) -> dict[str, Any]:
    """Load and resolve an observation YAML inheritance tree."""
    try:
        config, _ = resolve_observation_config(config_path)
    except (ObservationConfigError, FileNotFoundError) as error:
        raise ValueError(str(error)) from error
    return config


def load_catalogue(catalogue_path: str | Path) -> Table:
    """Load an explicitly selected catalogue file."""
    path = Path(catalogue_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Catalogue does not exist: {path}")
    return Table.read(path, unit_parse_strict="silent")


def load_reference_observation(
    observation_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate a prepared reference observation NPZ."""
    path = Path(observation_path).expanduser()
    with np.load(path, allow_pickle=False) as observation:
        keys = set(observation.files)
        expected_keys = {"x0", "mask"}
        if keys != expected_keys:
            raise ValueError(
                f"Reference observation must contain exactly {sorted(expected_keys)}; "
                f"found {sorted(keys)} in {path}."
            )
        raw_data = observation["x0"]
        raw_mask = observation["mask"]

    if not np.issubdtype(raw_data.dtype, np.floating):
        raise ValueError(
            f"Reference x0 must have a floating dtype; got {raw_data.dtype}."
        )
    if raw_mask.dtype != np.bool_:
        raise ValueError(
            f"Reference mask must have boolean dtype; got {raw_mask.dtype}."
        )

    data = np.asarray(raw_data, dtype=np.float32)
    data_mask = np.asarray(raw_mask, dtype=np.bool_)
    if data.ndim != 1 or data_mask.ndim != 1:
        raise ValueError("Reference data and mask must be one-dimensional.")
    if data.shape != data_mask.shape:
        raise ValueError(
            "Reference data and mask must have identical shapes; "
            f"got {data.shape} and {data_mask.shape}."
        )
    return data, data_mask


def build_mask(
    nside: int,
    *,
    galactic_plane_width_deg: float = 5,
    north_equatorial_pole_radius_deg: float = 42,
    default_a_team_radius_deg: float = 2,
    source_radii_deg: dict[str, float] | None = None,
) -> np.ndarray:
    """Build the equatorial RACS analysis mask in NEST ordering."""
    if source_radii_deg is None:
        source_radii_deg = DEFAULT_SOURCE_RADII_DEG

    masker = Masker(
        np.ones(hp.nside2npix(nside)),
        coordinate_system="equatorial",
    )
    masker.mask_galactic_plane(galactic_plane_width_deg)
    masker.mask_a_team_sources(radius_deg=default_a_team_radius_deg)
    masker.mask_equatorial_poles(
        north_radius=north_equatorial_pole_radius_deg,
    )
    for source_name, radius_deg in source_radii_deg.items():
        masker.mask_a_team_sources(
            radius_deg=radius_deg,
            source_names=[source_name],
        )

    return hp.reorder(masker.get_mask_map(), r2n=True)


def build_mask_from_observation_config(
    observation_config: Mapping[str, Any],
) -> np.ndarray:
    """Build the native mask declared by an observation configuration.

    A mask loaded from ``mask.config`` and masks described by the remaining
    top-level properties are combined so that a pixel is retained only when
    every configured mask retains it.
    """
    args = observation_config["args"]
    mask_args = observation_config["mask"]
    nside = args["nside"]

    mask_config = mask_args.get("config")
    legacy_required = {
        "galactic_plane_width_deg",
        "north_equatorial_pole_radius_deg",
        "default_a_team_radius_deg",
    }
    if mask_config is None and legacy_required <= mask_args.keys():
        return build_mask(
            nside,
            galactic_plane_width_deg=mask_args["galactic_plane_width_deg"],
            north_equatorial_pole_radius_deg=(
                mask_args["north_equatorial_pole_radius_deg"]
            ),
            default_a_team_radius_deg=mask_args["default_a_team_radius_deg"],
            source_radii_deg=dict(mask_args.get("source_radii_deg", {})),
        )

    combined_mask = np.ones(hp.nside2npix(nside), dtype=np.bool_)
    if mask_config is not None:
        configured_mask = yaml_to_mask(
            mask_config,
            coordinates="equatorial",
            nside=nside,
            ordering="NESTED",
        )
        combined_mask &= np.asarray(configured_mask, dtype=np.bool_)

    masker = Masker(
        np.ones(hp.nside2npix(nside)),
        coordinate_system="equatorial",
    )
    has_additional_mask = False
    if "galactic_plane_width_deg" in mask_args:
        masker.mask_galactic_plane(mask_args["galactic_plane_width_deg"])
        has_additional_mask = True
    if "default_a_team_radius_deg" in mask_args:
        masker.mask_a_team_sources(
            radius_deg=mask_args["default_a_team_radius_deg"]
        )
        has_additional_mask = True
    if "north_equatorial_pole_radius_deg" in mask_args:
        masker.mask_equatorial_poles(
            north_radius=mask_args["north_equatorial_pole_radius_deg"]
        )
        has_additional_mask = True
    for source_name, radius_deg in mask_args.get("source_radii_deg", {}).items():
        masker.mask_a_team_sources(
            radius_deg=radius_deg,
            source_names=[source_name],
        )
        has_additional_mask = True

    if has_additional_mask:
        combined_mask &= hp.reorder(masker.get_mask_map(), r2n=True).astype(
            np.bool_, copy=False
        )
    return combined_mask


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


def _catalogue_view(
    catalogue: Table,
    product: RacsProductSpec,
    minimum_flux: float,
    local_source_crossmatch_radius_arcsec: float | None,
) -> CatalogueToMap:
    view = CatalogueToMap(catalogue.copy())
    view.make_cut(
        product.columns.total_flux,
        minimum=minimum_flux,
        maximum=None,
    )

    if local_source_crossmatch_radius_arcsec is None:
        return view
    if (
        not np.isfinite(local_source_crossmatch_radius_arcsec)
        or local_source_crossmatch_radius_arcsec <= 0
    ):
        raise ValueError(
            "local_source_crossmatch_radius_arcsec must be positive and finite."
        )

    source_name_column = product.columns.source_name
    if source_name_column not in catalogue.colnames:
        raise ValueError(
            "Local-source cross-matching was requested, but catalogue column "
            f"{source_name_column!r} is missing."
        )
    view.crossmatch_local_sources(
        "equatorial",
        radius=local_source_crossmatch_radius_arcsec,
        source_name_A_column=source_name_column,
    )
    return view


def build_real_sample(
    model: Racs | RacsJax,
    catalogue: Table,
    flux_min: float,
    summary_features: list[SummaryFeature] | None = None,
    *,
    local_source_crossmatch_radius_arcsec: float | None,
    flux_temperature_min_mjy: float | None = None,
    flux_temperature_n_bins: int = DEFAULT_FLUX_TEMPERATURE_N_BINS,
    flux_temperature_quantiles: tuple[float, ...] = DEFAULT_FLUX_TEMPERATURE_QUANTILES,
    flux_elevation_n_bins: int = DEFAULT_FLUX_ELEVATION_N_BINS,
    flux_elevation_quantiles: tuple[float, ...] = DEFAULT_FLUX_ELEVATION_QUANTILES,
    save_map_plot: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    summary_features = list(summary_features or [])
    resolved_temperature_min = (
        flux_min if flux_temperature_min_mjy is None else flux_temperature_min_mjy
    )

    product = _model_product(model)

    map_catalogue = _catalogue_view(
        catalogue,
        product,
        flux_min,
        local_source_crossmatch_radius_arcsec,
    )
    density_map = map_catalogue.make_density_map(
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
        temperature_catalogue = _catalogue_view(
            catalogue,
            product,
            resolved_temperature_min,
            local_source_crossmatch_radius_arcsec,
        )
        flux, temperature = _real_catalogue_flux_temperature_samples(
            model,
            temperature_catalogue,
        )
    if "flux_elevation_quantiles" in summary_features:
        elevation_flux, elevation = _real_catalogue_flux_elevation_samples(
            model,
            map_catalogue,
        )
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
                        flux_min_mjy=resolved_temperature_min,
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
