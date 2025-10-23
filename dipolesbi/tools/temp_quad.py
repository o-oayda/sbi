from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import healpy as hp
import matplotlib

MPL_CACHE_DIR = Path(__file__).resolve().parent / "_mplcache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from dipolesbi.tools.utils import convert_to_l_dash


# Use TeX rendering for text labels
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

from matplotlib import texmanager as _texmanager

TEX_CACHE_DIR = MPL_CACHE_DIR / "tex.cache"
TEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_texmanager.TexManager.texcache = str(TEX_CACHE_DIR)


def scattomap(lon_deg: np.ndarray, lat_deg: np.ndarray, nside: int) -> np.ndarray:
    """Minimal helper approximating the behavior of the original scattomap."""
    colat = np.pi / 2 - np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    pix = hp.ang2pix(nside, colat, lon, nest=False)
    counts = np.bincount(pix, minlength=hp.nside2npix(nside)).astype(float)
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def build_quadrant_patch(ax: plt.Axes) -> PathPatch:
    """Return a PathPatch tracing the ideal top-right quadrant boundary of the Mollweide ellipse."""
    t_vals = np.linspace(0.0, np.pi / 2.0, 1024)
    x_vals = np.pi * np.cos(t_vals)
    y_vals = (np.pi / 2.0) * np.sin(t_vals)
    arc_points = list(zip(x_vals[::-1], y_vals[::-1]))
    verts = [(0.0, 0.0), (0.0, np.pi / 2.0)] + arc_points + [(np.pi, 0.0), (0.0, 0.0)]

    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    path = Path(verts, codes)
    return PathPatch(
        path,
        transform=ax.transData,
        facecolor="none",
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
    )


def main() -> None:
    rng = np.random.default_rng(123)
    nside = 64
    n_samples = 5000

    lon_samples = (
        np.rad2deg(rng.vonmises(mu=np.deg2rad(220), kappa=4, size=n_samples)) + 360
    ) % 360
    lat_samples = rng.normal(loc=45.0, scale=12.0, size=n_samples)

    shm = scattomap(lon_samples, lat_samples, nside=nside)
    shm_hires = scattomap(lon_samples, lat_samples, nside=4 * nside)
    smoothed = hp.sphtfunc.smoothing(shm_hires, sigma=np.deg2rad(5.0), verbose=False)

    proj_map = hp.mollview(smoothed, return_projected_map=True, xsize=1600)
    plt.close()

    xx = np.linspace(-np.pi, np.pi, proj_map.shape[1])
    yy = np.linspace(-np.pi / 2, np.pi / 2, proj_map.shape[0])
    Xa, Ya = np.meshgrid(xx, yy)

    proj = hp.projector.MollweideProj()

    peak_pix = int(np.argmax(shm))
    theta_peak, phi_peak = hp.pix2ang(nside, peak_pix, nest=False)
    peak_lon = np.rad2deg(phi_peak)
    peak_lat = 90.0 - np.rad2deg(theta_peak)

    markers = [
        ("CMB dipole", 263.99, 48.26, "k*"),
        ("Oxford dipole", 238.2, 28.8, "g*"),
        ("Posterior peak", peak_lon, peak_lat, "r*"),
    ]

    # Full-sky reference plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(
        proj_map,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        cmap="YlGnBu",
    )
    ax.contour(Xa, Ya, proj_map, colors="k", levels=3)
    def project_lonlat(lon_deg, lat_deg):
        lon_arr = np.atleast_1d(lon_deg).tolist()
        lat_arr = np.atleast_1d(lat_deg).tolist()
        x_proj, y_proj = proj.ang2xy(lon_arr, lat_arr, lonlat=True)
        x_arr = (np.pi / 2.0) * np.asarray(x_proj)
        y_arr = (np.pi / 2.0) * np.asarray(y_proj)
        x_arr = np.where(x_arr < 0, x_arr + 2 * np.pi, x_arr)
        return x_arr, y_arr

    for label, lon, lat, marker in markers:
        x_vals, y_vals = project_lonlat(lon, lat)
        ax.plot(x_vals, y_vals, marker, ms=12, linestyle="None", label=label)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi / 2, np.pi / 2)
    ax.set_title("Full Mollweide projection (reference)")
    ax.legend(loc="lower center", ncol=len(markers))
    plt.savefig("example_full.png", dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.savefig("example_full.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    # Top-right quadrant with curved boundary
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(
        proj_map,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        cmap="YlGnBu",
    )
    contour = ax.contour(Xa, Ya, proj_map, colors="k", levels=3)
    for label, lon, lat, marker in markers:
        x_vals, y_vals = project_lonlat(lon, lat)
        ax.plot(x_vals, y_vals, marker, ms=14, linestyle="None", label=label)

    ax.set_xlim(0.0, np.pi)
    ax.set_ylim(0.0, np.pi / 2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Configure ticks to use Galactic coordinates
    longitude_ticks = [330, 300, 270, 240, 210]
    xtick_positions, _ = project_lonlat(longitude_ticks, [0] * len(longitude_ticks))
    xtick_labels = []
    for val in longitude_ticks:
        if val == 330:
            xtick_labels.append(r"$\ell = 330^\circ$")
        else:
            xtick_labels.append(rf"${val}^\circ$")
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)

    latitude_ticks = [0, 30, 60]
    _, ytick_positions = project_lonlat([330] * len(latitude_ticks), latitude_ticks)
    ytick_labels = []
    for val in latitude_ticks:
        if val == 30:
            ytick_labels.append(r"$b = 30^\circ$")
        else:
            ytick_labels.append(rf"${val}^\circ$")
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, rotation=90, va="center")
    ax.tick_params(axis="both", which="both", length=0)

    # Hide default rectangular frame and replace with quadrant patch
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("none")

    boundary_patch = build_quadrant_patch(ax)
    ax.add_patch(boundary_patch)

    clip_patch = build_quadrant_patch(ax)
    clip_patch.set_edgecolor("none")
    im.set_clip_path(clip_patch)
    contour_collections = getattr(contour, "collections", None)
    if contour_collections is None:
        contour_iterable = contour if isinstance(contour, (list, tuple)) else [contour]
    else:
        contour_iterable = contour_collections
    for coll in contour_iterable:
        coll.set_clip_path(clip_patch)

    # Draw graticule lines for Galactic meridians/parallels
    for lat_deg in latitude_ticks:
        lon_curve = np.linspace(180.0, 360.0, 720)
        x_vals, y_vals = project_lonlat(lon_curve, np.full_like(lon_curve, lat_deg))
        mask = (x_vals >= 0) & (x_vals <= np.pi) & (y_vals >= 0) & (y_vals <= np.pi / 2)
        line, = ax.plot(
            x_vals[mask],
            y_vals[mask],
            linestyle=":",
            linewidth=0.7,
            color="black",
            alpha=0.5,
        )
        line.set_clip_path(clip_patch)

    for lon_deg in longitude_ticks:
        lat_curve = np.linspace(0.0, 90.0, 360)
        x_vals, y_vals = project_lonlat(np.full_like(lat_curve, lon_deg), lat_curve)
        mask = (x_vals >= 0) & (x_vals <= np.pi) & (y_vals >= 0) & (y_vals <= np.pi / 2)
        line, = ax.plot(
            x_vals[mask],
            y_vals[mask],
            linestyle=":",
            linewidth=0.7,
            color="black",
            alpha=0.5,
        )
        line.set_clip_path(clip_patch)

    clip_contour = build_quadrant_patch(ax)
    clip_contour.set_edgecolor("none")
    for coll in contour_iterable:
        coll.set_clip_path(clip_contour)
        coll.set_linewidth(0.7)

    plt.savefig("example_top_quad.png", dpi=200, bbox_inches="tight", pad_inches=0.0)
    plt.savefig("example_top_quad.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

    print("Saved example_full.png and example_top_quad.png")


if __name__ == "__main__":
    main()
