import argparse
import os
from typing import Tuple, List

from astropy.table import Table
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib import ticker
from matplotlib.colors import Normalize, LogNorm
from scipy.stats import binned_statistic_2d

from catsim import Catwise, CatwiseConfig


def median_error_grid(
    magnitudes: np.ndarray,
    log_coverages: np.ndarray,
    errors: np.ndarray,
    mag_edges: np.ndarray,
    cov_edges: np.ndarray,
    min_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    statistic, mag_edges, cov_edges, _ = binned_statistic_2d(
        magnitudes,
        log_coverages,
        errors,
        statistic="median",
        bins=[mag_edges, cov_edges],
    )
    counts, *_ = binned_statistic_2d(
        magnitudes,
        log_coverages,
        errors,
        statistic="count",
        bins=[mag_edges, cov_edges],
    )
    statistic[counts < min_count] = np.nan
    return statistic, mag_edges, cov_edges, counts


def extract_catalogue(
    mask_north_ecliptic: bool,
    cat_w1_max: float,
    cat_w12_min: float,
    mag_limits: Tuple[float, float],
    log_cov_limits: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = CatwiseConfig(
        cat_w1_max=cat_w1_max,
        cat_w12_min=cat_w12_min,
        magnitude_error_dist="gaussian",
    )
    catwise = Catwise(config)
    catwise.determine_masked_pixels(mask_north_ecliptic=mask_north_ecliptic)
    # catwise.load_catalogue()

    file_path = f'/home/oliver/Documents/catsim/src/catsim/data/catwise2020_corr_w120p5_w117p0.fits'
    print('Loading CatWISE2020...')
    catalogue = Table.read(file_path, unit_parse_strict='silent') # supress unit warning printouts
    print('Finished loading CatWISE2020.')

    pixel_idx = hp.ang2pix(
        catwise.nside,
        catalogue["l"],
        catalogue["b"],
        lonlat=True,
        nest=True,
    )
    keep = catwise.mask_map[pixel_idx] == 0

    m = np.asarray(catalogue["w1"], dtype=float)[keep]
    err = np.asarray(catalogue["w1e"], dtype=float)[keep]
    cov = np.asarray(catalogue["w1cov"], dtype=float)[keep]

    valid = (
        np.isfinite(m)
        & np.isfinite(err)
        & np.isfinite(cov)
        & (cov > 0)
        & (m >= mag_limits[0])
        & (m <= mag_limits[1])
    )
    m = m[valid]
    err = err[valid]
    cov = cov[valid]
    log_cov = np.log10(cov)

    cov_valid = (log_cov >= log_cov_limits[0]) & (log_cov <= log_cov_limits[1])
    m = m[cov_valid]
    err = err[cov_valid]
    log_cov = log_cov[cov_valid]

    return m, log_cov, err


def plot_panel(
    ax: plt.Axes,
    mag_edges: np.ndarray,
    log_cov_edges: np.ndarray,
    grid: np.ndarray,
    *,
    vline: float,
    cmap_norm: Normalize,
) -> None:
    extent = [mag_edges[0], mag_edges[-1], log_cov_edges[0], log_cov_edges[-1]]
    mesh = ax.imshow(
        grid.T,
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=cmap_norm,
        interpolation="nearest",
        aspect="auto",
    )
    if vline is not None:
        ax.axvline(vline, linestyle="--", color="black", linewidth=1)
    ax.set_xlim(mag_edges[0], mag_edges[-1])
    ax.set_ylim(log_cov_edges[0], log_cov_edges[-1])
    ax.set_xlabel("W1 magnitude")
    return mesh


def main() -> None:
    rcParams["text.usetex"] = True

    parser = argparse.ArgumentParser(
        description="Median raw W1 error vs. W1 magnitude and coverage."
    )
    parser.add_argument("--save-plot", action="store_true", help="Save figure instead of showing.")
    parser.add_argument("--figure-name", default="median_error_heatmap.pdf")
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--vline-mag", type=float, default=16.4, nargs="?")
    parser.add_argument("--mag-min", type=float, default=9.0)
    parser.add_argument("--mag-max", type=float, default=17.0)
    parser.add_argument("--mag-bins", type=int, default=100)
    parser.add_argument("--logcov-min", type=float, default=1.5)
    parser.add_argument("--logcov-max", type=float, default=4.0)
    parser.add_argument("--logcov-bins", type=int, default=100)
    parser.add_argument("--cat-w1-max", type=float, default=17.0)
    parser.add_argument("--cat-w12-min", type=float, default=0.5)
    parser.add_argument("--log-color", action="store_true", help="Use logarithmic colour scale.")
    args = parser.parse_args()

    mag_edges = np.linspace(args.mag_min, args.mag_max, args.mag_bins + 1)
    log_cov_edges = np.linspace(args.logcov_min, args.logcov_max, args.logcov_bins + 1)

    panels: List[np.ndarray] = []
    for mask_flag in (True, False):
        mags, log_covs, errs = extract_catalogue(
            mask_flag,
            args.cat_w1_max,
            args.cat_w12_min,
            (args.mag_min, args.mag_max),
            (args.logcov_min, args.logcov_max),
        )
        if mags.size == 0:
            grid = np.full((args.mag_bins, args.logcov_bins), np.nan)
        else:
            grid, _, _, _ = median_error_grid(
                mags,
                log_covs,
                errs,
                mag_edges,
                log_cov_edges,
                args.min_count,
            )
        panels.append(grid)

    stacked = np.stack([panel for panel in panels if not np.isnan(panel).all()])
    vmin = np.nanmin(stacked)
    vmax = np.nanmax(stacked)
    if args.log_color:
        positive_vmin = max(vmin, np.nextafter(0, 1))
        if positive_vmin >= vmax:
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = LogNorm(vmin=positive_vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    combined_mask = np.any(np.isfinite(stacked), axis=0)
    valid_cov = combined_mask.any(axis=0)
    if not valid_cov.any():
        y_min_lim, y_max_lim = args.logcov_min, args.logcov_max
    else:
        first_cov_bin = int(np.argmax(valid_cov))
        last_cov_bin = int(len(valid_cov) - np.argmax(valid_cov[::-1]))
        y_min_lim = log_cov_edges[first_cov_bin]
        y_max_lim = log_cov_edges[last_cov_bin]
        buffer = 0.02 * (y_max_lim - y_min_lim if y_max_lim > y_min_lim else 1.0)
        y_min_lim = max(args.logcov_min, y_min_lim - buffer)
        y_max_lim = min(args.logcov_max, y_max_lim + buffer)

    fig, axes = plt.subplots(2, 1, figsize=(3, 5), sharex=True, sharey=True)
    for idx, (ax, grid) in enumerate(zip(axes, panels[::-1])):  # top: unmasked
        mesh = plot_panel(
            ax,
            mag_edges,
            log_cov_edges,
            grid,
            vline=args.vline_mag,
            cmap_norm=norm,
        )
        ax.set_ylabel(r"$\log_{10} C_{W1}$")

    xticks = np.arange(np.ceil(args.mag_min), np.floor(args.mag_max) + 1)
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels([f"{tick:.0f}" for tick in xticks])
    axes[0].tick_params(labelbottom=False)

    yticks = np.linspace(y_min_lim, y_max_lim, 7)
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels([f"{tick:.1f}" for tick in yticks])
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels([f"{tick:.1f}" for tick in yticks])
    axes[1].tick_params(which="both", direction="out")
    axes[0].tick_params(which="both", direction="out")
    top_yticks = axes[0].get_yticklabels()
    bottom_yticks = axes[1].get_yticklabels()
    if top_yticks:
        top_yticks[0].set_visible(False)
    if bottom_yticks:
        bottom_yticks[-1].set_visible(False)

    axes[0].set_xlim(args.mag_min, args.mag_max)
    axes[1].set_xlim(args.mag_min, args.mag_max)
    axes[0].set_ylim(y_min_lim, y_max_lim)
    axes[1].set_ylim(y_min_lim, y_max_lim)

    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.1, top=0.95, hspace=0.0)
    cax = fig.add_axes([0.12, 0.955, 0.84, 0.03])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=8)

    # because matplotlib sucks major ass, I have to do this bullshit
    ticks = [0.012, 0.02, 0.03, 0.04, 0.05, 0.06]
    tick_labels = [str(tick) for tick in ticks]
    cbar.set_ticks(ticks=ticks, labels=tick_labels)

    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_label("Median W1 error", labelpad=4)

    if args.save_plot:
        out_dir = os.path.join(
            os.path.expanduser("~"),
            "Documents",
            "papers",
            "catwise_sbi",
            "figures",
        )
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, args.figure_name)
        print(f"Saving median error heatmap to {path}...")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
