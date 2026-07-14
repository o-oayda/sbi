from typing import Any

from catsim import (
    Racs,
    binned_flux_quantile_ndim,
    binned_flux_quantiles_exact,
    binned_flux_quantiles_histogram,
    empirical_bin_edges,
)
from dipoleutils.utils.samples import CatalogueToMap
import healpy as hp
import numpy as np


def _model_product(model: Any):
    product = getattr(model, "product", None)
    if product is not None:
        return product
    return model.cfg.product


def get_valid_native_counts(
    native_map: np.ndarray,
    native_mask: np.ndarray,
) -> np.ndarray:
    if native_map.shape != native_mask.shape:
        raise ValueError("native_map and native_mask must have matching shapes.")

    valid = native_mask.astype(bool, copy=False) & np.isfinite(native_map)
    if not np.any(valid):
        raise ValueError("Cannot build native-count summary with no unmasked native pixels.")

    counts = np.asarray(native_map[valid])
    rounded_counts = np.rint(counts)
    if np.any(rounded_counts < 0):
        raise ValueError("Native-count summary received negative counts.")
    return rounded_counts.astype(np.int64)


def _native_count_log_dispersion_feature(
    native_map: np.ndarray,
    native_mask: np.ndarray,
) -> np.ndarray:
    counts = get_valid_native_counts(native_map, native_mask).astype(np.float64)
    if counts.size < 2:
        raise ValueError("Native-count log dispersion requires at least two pixels.")
    mean = float(np.mean(counts))
    if mean <= 0.0:
        raise ValueError("Native-count log dispersion requires positive mean counts.")
    dispersion = float(np.var(counts, ddof=1) / mean)
    if dispersion <= 0.0:
        raise ValueError("Native-count log dispersion requires positive dispersion.")
    return np.asarray([np.log(dispersion)], dtype=np.float32)


def _flux_temperature_edges(
    model: Racs,
    n_bins: int,
) -> np.ndarray:
    temperatures = getattr(model, "tile_temperature_by_index", None)
    if temperatures is None:
        raise ValueError(
            "Flux-temperature summary requires model.tile_temperature_by_index."
        )
    return empirical_bin_edges(
        temperatures,
        n_bins,
        value_name="Flux-temperature",
    )


def _flux_elevation_edges(
    model: Racs,
    n_bins: int,
) -> np.ndarray:
    elevations = getattr(model, "elevation_lookup_values", None)
    if elevations is None:
        raise ValueError(
            "Flux-elevation summary requires model.elevation_lookup_values."
        )
    return empirical_bin_edges(
        elevations,
        n_bins,
        value_name="Flux-elevation",
    )


def _real_catalogue_flux_temperature_samples(
    model: Racs,
    c2map: CatalogueToMap,
) -> tuple[np.ndarray, np.ndarray]:
    product = _model_product(model)
    if getattr(model, "tile_temperature_by_index", None) is None:
        raise ValueError(
            "Flux-temperature summary requires model.tile_temperature_by_index."
        )
    if not hasattr(model, "_tile_index_from_sbid"):
        raise ValueError("Flux-temperature summary requires Racs tile metadata.")

    cut_catalogue = c2map.get_catalogue()
    ra = np.asarray(cut_catalogue[product.columns.ra], dtype=np.float64)
    dec = np.asarray(cut_catalogue[product.columns.dec], dtype=np.float64)
    flux = np.asarray(cut_catalogue[product.columns.total_flux], dtype=np.float64)
    sbid = np.asarray(cut_catalogue[product.columns.tile_id], dtype=np.int64)

    pixel_indices = hp.ang2pix(model.nside, ra, dec, lonlat=True, nest=True)
    in_mask = model.mask_map[pixel_indices].astype(bool, copy=False)
    if not np.any(in_mask):
        raise ValueError("Flux-temperature real summary has no sources in the mask.")

    tile_indices = np.full(sbid.shape, -1, dtype=np.int32)
    for idx, source_sbid in enumerate(sbid):
        tile_indices[idx] = model._tile_index_from_sbid.get(int(source_sbid), -1)
    valid_tile = tile_indices >= 0
    temperatures = np.full(flux.shape, np.nan, dtype=np.float64)
    temperatures[valid_tile] = np.asarray(
        model.tile_temperature_by_index,
        dtype=np.float64,
    )[tile_indices[valid_tile]]

    keep = in_mask & valid_tile & np.isfinite(flux) & np.isfinite(temperatures)
    if not np.any(keep):
        raise ValueError(
            "Flux-temperature real summary has no finite retained flux/temperature pairs."
        )
    return flux[keep], temperatures[keep]


def _real_catalogue_flux_elevation_samples(
    model: Racs,
    c2map: CatalogueToMap,
) -> tuple[np.ndarray, np.ndarray]:
    product = _model_product(model)
    elevation_column = product.columns.elevation
    if elevation_column is None:
        raise ValueError(
            f"{product.label} does not define an elevation column; "
            "flux-elevation summary requires catalogue ALT data."
        )

    cut_catalogue = c2map.get_catalogue()
    if elevation_column not in cut_catalogue.colnames:
        raise ValueError(
            f"{product.label} catalogue is missing elevation column "
            f"{elevation_column!r}."
        )
    ra = np.asarray(cut_catalogue[product.columns.ra], dtype=np.float64)
    dec = np.asarray(cut_catalogue[product.columns.dec], dtype=np.float64)
    flux = np.asarray(cut_catalogue[product.columns.total_flux], dtype=np.float64)
    elevation = np.asarray(cut_catalogue[elevation_column], dtype=np.float64)

    pixel_indices = hp.ang2pix(model.nside, ra, dec, lonlat=True, nest=True)
    in_mask = model.mask_map[pixel_indices].astype(bool, copy=False)
    keep = in_mask & np.isfinite(flux) & np.isfinite(elevation)
    if not np.any(keep):
        raise ValueError(
            "Flux-elevation real summary has no finite retained flux/elevation pairs."
        )
    return flux[keep], elevation[keep]


def _flux_temperature_quantile_features(
    observed_flux: np.ndarray,
    temperature: np.ndarray,
    temp_edges: np.ndarray,
    quantiles: tuple[float, ...],
) -> np.ndarray:
    return binned_flux_quantiles_exact(
        observed_flux,
        temperature,
        bin_edges=temp_edges,
        quantiles=quantiles,
        value_name="temperature",
    )


def _flux_elevation_quantile_features(
    observed_flux: np.ndarray,
    elevation: np.ndarray,
    elevation_edges: np.ndarray,
    quantiles: tuple[float, ...],
) -> np.ndarray:
    return binned_flux_quantiles_exact(
        observed_flux,
        elevation,
        bin_edges=elevation_edges,
        quantiles=quantiles,
        value_name="elevation",
    )


def _flux_binned_histogram_quantile_features(
    observed_flux: np.ndarray,
    bin_values: np.ndarray,
    *,
    bin_edges: np.ndarray,
    quantiles: tuple[float, ...],
    flux_min_mjy: float,
    flux_max_mjy: float,
    n_flux_bins: int = 128,
    empty_value: float = 0.0,
) -> np.ndarray:
    return binned_flux_quantiles_histogram(
        observed_flux,
        bin_values,
        bin_edges=bin_edges,
        quantiles=quantiles,
        flux_min_mjy=flux_min_mjy,
        flux_max_mjy=flux_max_mjy,
        n_flux_bins=n_flux_bins,
        empty_value=empty_value,
    )


def _flux_temperature_histogram_quantile_features(
    observed_flux: np.ndarray,
    temperature: np.ndarray,
    *,
    temp_edges: np.ndarray,
    quantiles: tuple[float, ...],
    flux_min_mjy: float,
    flux_max_mjy: float,
    n_flux_bins: int = 128,
    empty_value: float = 0.0,
) -> np.ndarray:
    return _flux_binned_histogram_quantile_features(
        observed_flux,
        temperature,
        bin_edges=temp_edges,
        quantiles=quantiles,
        flux_min_mjy=flux_min_mjy,
        flux_max_mjy=flux_max_mjy,
        n_flux_bins=n_flux_bins,
        empty_value=empty_value,
    )


def _flux_elevation_histogram_quantile_features(
    observed_flux: np.ndarray,
    elevation: np.ndarray,
    *,
    elevation_edges: np.ndarray,
    quantiles: tuple[float, ...],
    flux_min_mjy: float,
    flux_max_mjy: float,
    n_flux_bins: int = 128,
    empty_value: float = 0.0,
) -> np.ndarray:
    return _flux_binned_histogram_quantile_features(
        observed_flux,
        elevation,
        bin_edges=elevation_edges,
        quantiles=quantiles,
        flux_min_mjy=flux_min_mjy,
        flux_max_mjy=flux_max_mjy,
        n_flux_bins=n_flux_bins,
        empty_value=empty_value,
    )


def _flux_temperature_quantile_ndim(
    n_temp_bins: int,
    quantiles: tuple[float, ...],
) -> int:
    return binned_flux_quantile_ndim(n_temp_bins, quantiles)


def _flux_elevation_quantile_ndim(
    n_elevation_bins: int,
    quantiles: tuple[float, ...],
) -> int:
    return binned_flux_quantile_ndim(n_elevation_bins, quantiles)
