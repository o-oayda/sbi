from typing import Any

from catsim import Racs
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
    if n_bins < 1:
        raise ValueError("flux-temperature summary requires at least one bin.")
    temperatures = getattr(model, "tile_temperature_by_index", None)
    if temperatures is None:
        raise ValueError(
            "Flux-temperature summary requires model.tile_temperature_by_index."
        )
    finite = np.asarray(temperatures, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Flux-temperature summary found no finite tile temperatures.")
    temp_min = float(np.min(finite))
    temp_max = float(np.max(finite))
    if not temp_min < temp_max:
        raise ValueError(
            "Flux-temperature summary requires a non-zero temperature range."
        )
    return np.linspace(temp_min, temp_max, n_bins + 1, dtype=np.float64)


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


def _flux_temperature_quantile_features(
    observed_flux: np.ndarray,
    temperature: np.ndarray,
    temp_edges: np.ndarray,
    quantiles: tuple[float, ...],
) -> np.ndarray:
    observed_flux = np.asarray(observed_flux, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    temp_edges = np.asarray(temp_edges, dtype=np.float64)
    quantile_array = np.asarray(quantiles, dtype=np.float64)

    if observed_flux.shape != temperature.shape:
        raise ValueError("observed_flux and temperature must have matching shapes.")
    if temp_edges.ndim != 1 or temp_edges.size < 2:
        raise ValueError("temp_edges must be a one-dimensional array of bin edges.")
    if np.any(~np.isfinite(temp_edges)) or np.any(np.diff(temp_edges) <= 0):
        raise ValueError("temp_edges must be finite and strictly increasing.")
    if quantile_array.ndim != 1 or quantile_array.size == 0:
        raise ValueError("At least one flux quantile is required.")
    if np.any((quantile_array < 0.0) | (quantile_array > 1.0)):
        raise ValueError("Flux quantiles must lie in [0, 1].")

    valid = np.isfinite(observed_flux) & np.isfinite(temperature)
    flux = observed_flux[valid]
    temp = temperature[valid]
    if flux.size == 0:
        raise ValueError("Flux-temperature summary has no finite flux/temperature pairs.")

    features: list[float] = []
    for bin_idx, (lo, hi) in enumerate(zip(temp_edges[:-1], temp_edges[1:])):
        if bin_idx == temp_edges.size - 2:
            in_bin = (temp >= lo) & (temp <= hi)
        else:
            in_bin = (temp >= lo) & (temp < hi)
        if not np.any(in_bin):
            raise ValueError(
                "Flux-temperature summary has an empty temperature bin "
                f"[{lo:.6g}, {hi:.6g}{']' if bin_idx == temp_edges.size - 2 else ')'}."
            )
        features.extend(np.quantile(flux[in_bin], quantile_array))

    return np.asarray(features, dtype=np.float32)


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
    observed_flux = np.asarray(observed_flux, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    temp_edges = np.asarray(temp_edges, dtype=np.float64)
    quantile_array = np.asarray(quantiles, dtype=np.float64)

    if observed_flux.shape != temperature.shape:
        raise ValueError("observed_flux and temperature must have matching shapes.")
    if temp_edges.ndim != 1 or temp_edges.size < 2:
        raise ValueError("temp_edges must be a one-dimensional array of bin edges.")
    if np.any(~np.isfinite(temp_edges)) or np.any(np.diff(temp_edges) <= 0):
        raise ValueError("temp_edges must be finite and strictly increasing.")
    if quantile_array.ndim != 1 or quantile_array.size == 0:
        raise ValueError("At least one flux quantile is required.")
    if np.any((quantile_array < 0.0) | (quantile_array > 1.0)):
        raise ValueError("Flux quantiles must lie in [0, 1].")
    if not np.isfinite(flux_min_mjy) or flux_min_mjy <= 0.0:
        raise ValueError("flux_min_mjy must be positive and finite.")
    if not np.isfinite(flux_max_mjy) or flux_max_mjy <= flux_min_mjy:
        raise ValueError("flux_max_mjy must be finite and greater than flux_min_mjy.")
    if n_flux_bins < 1:
        raise ValueError("n_flux_bins must be at least 1.")

    valid = (
        np.isfinite(observed_flux)
        & np.isfinite(temperature)
        & (observed_flux >= flux_min_mjy)
    )
    flux = observed_flux[valid]
    temp = temperature[valid]
    if flux.size == 0:
        raise ValueError("Flux-temperature summary has no finite flux/temperature pairs.")

    z_max = np.log10(flux_max_mjy / flux_min_mjy)
    z = np.log10(np.maximum(flux, flux_min_mjy) / flux_min_mjy)
    z = np.clip(z, 0.0, np.nextafter(z_max, 0.0))
    z_edges = np.linspace(0.0, z_max, n_flux_bins + 1, dtype=np.float64)

    n_temp_bins = temp_edges.size - 1
    hist = np.zeros((n_temp_bins, n_flux_bins), dtype=np.float64)
    for bin_idx, (lo, hi) in enumerate(zip(temp_edges[:-1], temp_edges[1:])):
        if bin_idx == n_temp_bins - 1:
            in_temp_bin = (temp >= lo) & (temp <= hi)
        else:
            in_temp_bin = (temp >= lo) & (temp < hi)
        if not np.any(in_temp_bin):
            continue
        flux_bins = np.searchsorted(z_edges, z[in_temp_bin], side="right") - 1
        flux_bins = np.clip(flux_bins, 0, n_flux_bins - 1)
        hist[bin_idx] = np.bincount(flux_bins, minlength=n_flux_bins)

    features: list[float] = []
    for row in hist:
        total = float(np.sum(row))
        if total <= 0.0:
            features.extend([empty_value] * quantile_array.size)
            continue
        cumulative = np.cumsum(row)
        for q in quantile_array:
            target = np.finfo(np.float64).eps if q <= 0.0 else q * total
            flux_bin = int(np.searchsorted(cumulative, target, side="left"))
            flux_bin = int(np.clip(flux_bin, 0, n_flux_bins - 1))
            previous_cdf = 0.0 if flux_bin == 0 else float(cumulative[flux_bin - 1])
            current_cdf = float(cumulative[flux_bin])
            denominator = max(current_cdf - previous_cdf, np.finfo(np.float64).eps)
            fraction = np.clip((target - previous_cdf) / denominator, 0.0, 1.0)
            z_quantile = z_edges[flux_bin] + fraction * (
                z_edges[flux_bin + 1] - z_edges[flux_bin]
            )
            features.append(float(flux_min_mjy * np.power(10.0, z_quantile)))

    return np.asarray(features, dtype=np.float32)


def _flux_temperature_quantile_ndim(
    n_temp_bins: int,
    quantiles: tuple[float, ...],
) -> int:
    return n_temp_bins * len(quantiles)
