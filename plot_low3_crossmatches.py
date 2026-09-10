#!/usr/bin/env python3
"""Plot matched RACS-LOW3 positions in equatorial and Galactic coordinates."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from dipoleutils.utils.crossmatch import CrossMatch
from dipoleutils.utils.samples import CatalogueToMap
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import yaml


DEFAULT_TABLE = Path(
    "derived/observations/racs_low3_flux15_ds4/local_source_crossmatches.fits"
)
DEFAULT_OBSERVATION_CONFIG = Path(
    "workflow/configs/observations/racs_low3_flux15_ds4.yaml"
)
DEFAULT_SITE_CONFIG = Path("workflow/configs/sites/mac.yaml")
NSIDE = 64
PIXEL_AREA_DEG2 = hp.nside2pixarea(NSIDE, degrees=True)
DECLINATION_BIN_WIDTH_DEG = 5
DEFAULT_BOOTSTRAP_SAMPLES = 2_000


def make_pixel_counts(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    source_pixels = hp.ang2pix(NSIDE, ra_deg, dec_deg, lonlat=True, nest=True)
    return np.bincount(source_pixels, minlength=hp.nside2npix(NSIDE))


def declination_density(
    pixel_counts: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, pixel_declination = hp.pix2ang(
        NSIDE,
        np.arange(pixel_counts.size),
        lonlat=True,
        nest=True,
    )
    declination_edges = np.arange(
        -90,
        90 + DECLINATION_BIN_WIDTH_DEG,
        DECLINATION_BIN_WIDTH_DEG,
    )
    unmasked_declination = pixel_declination[mask]
    unmasked_counts = pixel_counts[mask]
    sources_per_bin, _ = np.histogram(
        unmasked_declination,
        bins=declination_edges,
        weights=unmasked_counts,
    )
    pixels_per_bin, _ = np.histogram(
        unmasked_declination,
        bins=declination_edges,
    )
    mean_counts_per_pixel = np.divide(
        sources_per_bin,
        pixels_per_bin,
        out=np.full(sources_per_bin.shape, np.nan),
        where=pixels_per_bin > 0,
    )
    return declination_edges, mean_counts_per_pixel / PIXEL_AREA_DEG2


def bootstrap_density_interval(
    pixel_counts: np.ndarray,
    mask: np.ndarray,
    declination_edges: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, pixel_declination = hp.pix2ang(
        NSIDE,
        np.arange(pixel_counts.size),
        lonlat=True,
        nest=True,
    )
    unmasked_declination = pixel_declination[mask]
    unmasked_counts = pixel_counts[mask]
    lower = np.full(declination_edges.size - 1, np.nan)
    upper = np.full(declination_edges.size - 1, np.nan)
    rng = np.random.default_rng(seed)

    for index, (lower_edge, upper_edge) in enumerate(
        zip(declination_edges[:-1], declination_edges[1:])
    ):
        in_bin = (unmasked_declination >= lower_edge) & (
            unmasked_declination < upper_edge
        )
        counts_in_bin = unmasked_counts[in_bin]
        if counts_in_bin.size == 0:
            continue
        bootstrap_density = rng.choice(
            counts_in_bin,
            size=(samples, counts_in_bin.size),
            replace=True,
        ).mean(axis=1) / PIXEL_AREA_DEG2
        lower[index], upper[index] = np.percentile(
            bootstrap_density,
            [16, 84],
        )
    return lower, upper


def mean_radius_by_declination(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    radius_arcsec: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    source_pixels = hp.ang2pix(
        NSIDE,
        ra_deg,
        dec_deg,
        lonlat=True,
        nest=True,
    )
    unmasked = mask[source_pixels]
    declination_edges = np.arange(
        -90,
        90 + DECLINATION_BIN_WIDTH_DEG,
        DECLINATION_BIN_WIDTH_DEG,
    )
    radius_sum, _ = np.histogram(
        dec_deg[unmasked],
        bins=declination_edges,
        weights=radius_arcsec[unmasked],
    )
    source_count, _ = np.histogram(
        dec_deg[unmasked],
        bins=declination_edges,
    )
    return np.divide(
        radius_sum,
        source_count,
        out=np.full(radius_sum.shape, np.nan),
        where=source_count > 0,
    )


def save_declination_comparison(
    dec_pixel_counts: np.ndarray,
    dec_corr_pixel_counts: np.ndarray,
    variable_radius_pixel_counts: np.ndarray,
    mean_radius_arcsec: np.ndarray,
    mask: np.ndarray,
    output: Path,
    *,
    bootstrap_samples: int,
    seed: int,
) -> None:
    declination_edges, dec_density = declination_density(dec_pixel_counts, mask)
    _, dec_corr_density = declination_density(dec_corr_pixel_counts, mask)
    _, variable_radius_density = declination_density(
        variable_radius_pixel_counts,
        mask,
    )
    variable_radius_lower, variable_radius_upper = bootstrap_density_interval(
        variable_radius_pixel_counts,
        mask,
        declination_edges,
        samples=bootstrap_samples,
        seed=seed,
    )

    figure, axis = plt.subplots(figsize=(7, 4))
    declination_centres = (
        declination_edges[:-1] + declination_edges[1:]
    ) / 2
    series = (
        (dec_density, "Dec", "C0"),
        (dec_corr_density, "Dec_corr, 5\"", "C1"),
        (
            variable_radius_density,
            r"Dec_corr, $0.5\,\mathrm{PSF}_{\mathrm{maj}}$",
            "C2",
        ),
    )
    for density, label, colour in series:
        axis.stairs(density, declination_edges, color=colour, linewidth=2, label=label)
        axis.stairs(
            density,
            declination_edges,
            color=colour,
            fill=True,
            alpha=0.15,
        )
    axis.errorbar(
        declination_centres,
        variable_radius_density,
        yerr=np.vstack(
            [
                variable_radius_density - variable_radius_lower,
                variable_radius_upper - variable_radius_density,
            ]
        ),
        fmt="none",
        ecolor="C2",
        elinewidth=0.8,
        capsize=1.5,
    )
    radius_axis = axis.twinx()
    radius_line = radius_axis.plot(
        declination_centres,
        mean_radius_arcsec,
        color="C3",
        linewidth=1.5,
        linestyle='--',
        label=r"$0.5\,\mathrm{PSF}_{\mathrm{maj}}$ radius",
    )[0]
    radius_axis.set_ylabel("Crossmatch radius (\")", color="C3")
    radius_axis.tick_params(axis="y", colors="C3")
    axis.set(
        xlabel=r"Declination $\delta^\circ$",
        ylabel=r"Count per square degree",
        # title=(
        #     "LOW3 crossmatch density versus declination "
        #     f"({DECLINATION_BIN_WIDTH_DEG} deg bins; unmasked pixels)"
        # ),
        xlim=(-90, 90),
    )
    density_handles, density_labels = axis.get_legend_handles_labels()
    axis.legend(
        [*density_handles, radius_line],
        [*density_labels, radius_line.get_label()],
        loc='upper left',
        ncols=2
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)
    print(f"Saved {output}")


def corrected_declination_matches(
    observation_config_path: Path,
    site_config_path: Path,
) -> tuple[Table, Table, Table]:
    with observation_config_path.open(encoding="utf-8") as stream:
        observation = yaml.safe_load(stream)
    with site_config_path.open(encoding="utf-8") as stream:
        site = yaml.safe_load(stream)

    catalogue_id = observation["datasets"]["catalogue"]
    catalogue_path = Path(site["data_locations"][catalogue_id]["path"])
    column_names = [
        "RA",
        "Dec",
        "Dec_corr",
        "Total_flux",
        "Name",
        "Source_ID",
        "PSF_Maj",
    ]
    with fits.open(catalogue_path, memmap=True) as hdus:
        catalogue = Table(
            {name: np.array(hdus[1].data[name]) for name in column_names}
        )
    catalogue = catalogue[catalogue["Total_flux"] >= observation["args"]["flux_min"]]

    catalogue_view = CatalogueToMap(catalogue)
    local_sources = catalogue_view._load_local_table()

    def match(radius_arcsec: float | np.ndarray) -> Table:
        crossmatcher = CrossMatch(catalogue, local_sources, "equatorial")
        crossmatcher.lonA_column = "RA"
        crossmatcher.latA_column = "Dec_corr"
        crossmatcher.cross_match(
            radius_arcsec,
            source_name_A_column="Name",
            source_name_B_column="source",
        )
        return crossmatcher.get_common_sources()

    fixed_radius_matches = match(
        observation["args"]["local_source_crossmatch_radius_arcsec"]
    )
    variable_radius_arcsec = 0.5 * np.asarray(catalogue["PSF_Maj"], dtype=float)
    invalid_radius = ~np.isfinite(variable_radius_arcsec) | (
        variable_radius_arcsec <= 0
    )
    if np.any(invalid_radius):
        raise ValueError(
            f"Found {invalid_radius.sum()} invalid PSF-derived crossmatch radii."
        )
    variable_radius_matches = match(variable_radius_arcsec)
    radius_sample = Table(
        {
            "RA": catalogue["RA"],
            "Dec_corr": catalogue["Dec_corr"],
            "radius_arcsec": variable_radius_arcsec,
        }
    )
    return fixed_radius_matches, variable_radius_matches, radius_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", nargs="?", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--mask", type=Path)
    parser.add_argument(
        "--observation-config",
        type=Path,
        default=DEFAULT_OBSERVATION_CONFIG,
    )
    parser.add_argument("--site-config", type=Path, default=DEFAULT_SITE_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--galactic-output", type=Path)
    parser.add_argument("--declination-output", type=Path)
    parser.add_argument("--healpix-output", type=Path)
    parser.add_argument("--dec-corr-only-output", type=Path)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")

    table = Table.read(args.table, format="fits")
    mask_path = args.mask or args.table.parent / "mask.npy"
    mask = np.load(mask_path, allow_pickle=False)
    if mask.dtype != np.bool_ or mask.ndim != 1:
        raise ValueError(f"Expected a one-dimensional boolean mask: {mask_path}")
    if hp.npix2nside(mask.size) != NSIDE:
        raise ValueError(f"Expected an NSIDE={NSIDE} mask: {mask_path}")

    ra = np.asarray(table["A_RA"], dtype=float)
    dec = np.asarray(table["A_Dec"], dtype=float)
    finite = np.isfinite(ra) & np.isfinite(dec)

    output = args.output or args.table.with_suffix(".png")
    galactic_output = args.galactic_output or output.with_name(
        f"{output.stem}_galactic{output.suffix}"
    )
    declination_output = args.declination_output or output.with_name(
        f"{output.stem}_density_vs_declination{output.suffix}"
    )
    healpix_output = args.healpix_output or output.with_name(
        f"{output.stem}_healpix{output.suffix}"
    )
    dec_corr_only_output = args.dec_corr_only_output or output.with_name(
        f"{output.stem}_dec_corr_only{output.suffix}"
    )
    coordinates = SkyCoord(ra=ra[finite] * u.deg, dec=dec[finite] * u.deg)

    def save_plot(longitude: Angle, latitude: Angle, frame: str, path: Path) -> None:
        # Wrap at 180 degrees and reverse longitude so east is left.
        plot_lon = -longitude.wrap_at(180 * u.deg).radian
        figure = plt.figure(figsize=(8, 4.5))
        axis = figure.add_subplot(111, projection="mollweide")
        axis.scatter(plot_lon, latitude.radian, s=5, alpha=0.65, linewidths=0)
        axis.grid(alpha=0.35)
        axis.set_title(f"{frame}")
        figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200)
        plt.close(figure)
        print(f"Saved {path}")

    save_plot(coordinates.ra, coordinates.dec, "equatorial", output)
    save_plot(
        coordinates.galactic.l,
        coordinates.galactic.b,
        "Galactic",
        galactic_output,
    )

    pixel_counts = make_pixel_counts(coordinates.ra.deg, coordinates.dec.deg)
    masked_density = pixel_counts.astype(float) / PIXEL_AREA_DEG2
    masked_density[~mask] = np.nan
    pixel_counts = pixel_counts.astype('float64')
    pixel_counts[~mask] = np.nan
    hp.projview(
        pixel_counts,
        nest=True,
        coord="C",
        graticule=True,
        graticule_labels=True,
        max=2,
        # title=f"LOW3 crossmatch density map (NSIDE={NSIDE}; unmasked pixels)",
        unit=r"Count per pixel",
    )
    healpix_output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(healpix_output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {healpix_output}")

    (
        dec_corr_matches,
        variable_radius_matches,
        variable_radius_sample,
    ) = corrected_declination_matches(args.observation_config, args.site_config)
    dec_corr_only = ~np.isin(
        np.asarray(dec_corr_matches["A_Source_ID"]),
        np.asarray(table["A_Source_ID"]),
    )
    dec_corr_only_coordinates = SkyCoord(
        ra=np.asarray(dec_corr_matches["A_RA"])[dec_corr_only] * u.deg,
        dec=np.asarray(dec_corr_matches["A_Dec_corr"])[dec_corr_only] * u.deg,
    )
    save_plot(
        dec_corr_only_coordinates.ra,
        dec_corr_only_coordinates.dec,
        "equatorial",
        dec_corr_only_output,
    )
    dec_corr_pixel_counts = make_pixel_counts(
        np.asarray(dec_corr_matches["A_RA"], dtype=float),
        np.asarray(dec_corr_matches["A_Dec_corr"], dtype=float),
    )
    variable_radius_pixel_counts = make_pixel_counts(
        np.asarray(variable_radius_matches["A_RA"], dtype=float),
        np.asarray(variable_radius_matches["A_Dec_corr"], dtype=float),
    )
    mean_radius_arcsec = mean_radius_by_declination(
        np.asarray(variable_radius_sample["RA"], dtype=float),
        np.asarray(variable_radius_sample["Dec_corr"], dtype=float),
        np.asarray(variable_radius_sample["radius_arcsec"], dtype=float),
        mask,
    )
    save_declination_comparison(
        pixel_counts,
        dec_corr_pixel_counts,
        variable_radius_pixel_counts,
        mean_radius_arcsec,
        mask,
        declination_output,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
