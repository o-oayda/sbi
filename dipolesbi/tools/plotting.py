from numpy.typing import NDArray
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Bbox
import numpy as np
from typing import Callable, Sequence, Literal

from dipolesbi.tools.maps import average_smooth_map
from dipolesbi.tools.utils import (
    compute_2D_contours,
    convert_to_l_dash,
    samples_to_hpmap,
)


SKY_PROBABILITY_COLOR_CYCLE: list[str] = [
    "#006FED",
    "#E03424",
    "#808080",
    "#009966",
    "#000866",
    "#336600",
    "#006633",
    "#FF00FF",
    "#FF0000",
]


marker_cycle = ["*", "+", ".", "h"]


def quad_tick_labels() -> tuple[list[str], list[str]]:
    """Return nice tick labels for the top-right sky quadrant."""
    y_labels: list[str] = []
    for idx in range(1, 6):
        value = -90 + 30 * idx
        if idx == 4:
            y_labels.append(r"$b=" + f"{value}" + r"^\circ$")
        else:
            y_labels.append(r"$" + f"{value}" + r"^\circ$")

    x_labels: list[str] = []
    for idx in range(1, 12):
        if idx < 6:
            value = -180 + 30 * idx
        if idx == 6:
            x_labels.append(r"$l=0^\circ$")
        else:
            value = 540 - 30 * idx
            x_labels.append(r"$" + f"{value}" + r"^\circ$")

    return x_labels, y_labels


def get_top_quadrant_bbox(
        ax: matplotlib.axes.Axes,
        fig: matplotlib.figure.Figure,
        plot_style: Literal['legacy', 'modern']
) -> Bbox:
    """Calculate the bounding box for the top-right quadrant of a Mollweide projection."""
    if plot_style == "legacy":
        x0, x1 = -0.25, np.pi + 0.45
        y0, y1 = -0.22, np.pi / 2
    elif plot_style == "modern":
        x0, x1 = -0.2, np.pi + 0.05
        y0, y1 = 0, np.pi / 2 - 0.1

    bbox = Bbox([[x0, y0], [x1, y1]])
    return bbox.transformed(ax.transData).transformed(fig.dpi_scale_trans.inverted())

def _get_mollweide_projector() -> hp.projector.MollweideProj:
    """Return a cached Mollweide projector."""
    projector = getattr(_get_mollweide_projector, "_cached", None)
    if projector is None:
        projector = hp.projector.MollweideProj()
        setattr(_get_mollweide_projector, "_cached", projector)
    return projector


_MOLLWEIDE_PROJECTOR = hp.projector.MollweideProj()


def _project_lonlat(
    lon_deg: Sequence[float] | float,
    lat_deg: Sequence[float] | float,
    *,
    wrap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Project Galactic lon/lat in degrees onto Mollweide x/y coordinates."""
    lon_arr = np.atleast_1d(lon_deg).astype(np.float64)
    lat_arr = np.atleast_1d(lat_deg).astype(np.float64)
    projector = _get_mollweide_projector()
    x_proj, y_proj = projector.ang2xy(lon_arr.tolist(), lat_arr.tolist(), lonlat=True)
    x_vals = (np.pi / 2.0) * np.asarray(x_proj, dtype=np.float64)
    y_vals = (np.pi / 2.0) * np.asarray(y_proj, dtype=np.float64)
    if wrap:
        x_vals = np.where(x_vals < 0.0, x_vals + 2.0 * np.pi, x_vals)
    x_vals = np.atleast_1d(x_vals)
    y_vals = np.atleast_1d(y_vals)
    return x_vals, y_vals


def _build_top_quadrant_patch(ax: matplotlib.axes.Axes) -> PathPatch:
    """Construct a patch tracing the true top-right Mollweide quadrant boundary."""
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


def _ensure_top_quadrant_setup(ax: matplotlib.axes.Axes) -> PathPatch:
    """Configure the current axes for the modern top-quadrant view."""
    patch = getattr(ax, "_sbi_top_quad_patch", None)
    if patch is None:
        patch = _build_top_quadrant_patch(ax)
        ax.add_patch(patch)
        ax._sbi_top_quad_patch = patch  # type: ignore[attr-defined]

        ax.patch.set_facecolor("white")
        ax.patch.set_edgecolor("none")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)

        longitude_ticks = [330, 300, 270, 240, 210]
        xtick_positions, _ = _project_lonlat(longitude_ticks, [0] * len(longitude_ticks), wrap=False)
        xtick_labels: list[str] = []
        for value in longitude_ticks:
            if value == 330:
                xtick_labels.append(r"$\ell = 330^\circ$")
            else:
                xtick_labels.append(rf"${value}^\circ$")
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)

        latitude_ticks = [0, 30, 60]
        _, ytick_positions = _project_lonlat([longitude_ticks[0]] * len(latitude_ticks), latitude_ticks, wrap=False)
        ytick_labels: list[str] = []
        for value in latitude_ticks:
            if value == 30:
                ytick_labels.append(r"$b = 30^\circ$")
            else:
                ytick_labels.append(rf"${value}^\circ$")
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, rotation=90, va="center")

        longitude_curve = np.linspace(0.0, 360.0, 1440)
        latitude_curve = np.linspace(0.0, 90.0, 720)
        graticule_lines: list[matplotlib.lines.Line2D] = []
        for lat_val in latitude_ticks:
            x_vals, y_vals = _project_lonlat(
                longitude_curve, np.full_like(longitude_curve, lat_val), wrap=False
            )
            mask = (x_vals >= 0.0) & (x_vals <= np.pi) & (y_vals >= 0.0) & (y_vals <= np.pi / 2.0)
            (line,) = ax.plot(
                x_vals[mask],
                y_vals[mask],
                linestyle=":",
                linewidth=0.7,
                color="black",
                alpha=0.5,
                zorder=4,
            )
            graticule_lines.append(line)

        for lon_deg in longitude_ticks:
            x_vals, y_vals = _project_lonlat(
                np.full_like(latitude_curve, lon_deg), latitude_curve, wrap=False
            )
            mask = (x_vals >= 0.0) & (x_vals <= np.pi) & (y_vals >= 0.0) & (y_vals <= np.pi / 2.0)
            (line,) = ax.plot(
                x_vals[mask],
                y_vals[mask],
                linestyle=":",
                linewidth=0.7,
                color="black",
                alpha=0.5,
                zorder=4,
            )
            graticule_lines.append(line)
        for line in graticule_lines:
            line.set_zorder(4)
    return getattr(ax, "_sbi_top_quad_patch")  # type: ignore[attr-defined]


def _clip_artists_to_patch(ax: matplotlib.axes.Axes, patch: PathPatch) -> None:
    """Clip existing axis artists against the quadrant patch."""
    for collection in list(ax.collections):
        if collection is patch:
            continue
        collection.set_clip_path(patch)
    for image in list(ax.images):
        image.set_clip_path(patch)
    for line in list(ax.lines):
        line.set_clip_path(patch)
    for other_patch in list(ax.patches):
        if other_patch is patch:
            continue
        other_patch.set_clip_path(patch)


def posterior_predictive_check(
        samples: NDArray,
        model: Callable,
        n_samples: int = 5,
        **projview_kwargs
) -> None:
    random_integers = np.random.randint(  # type: ignore[union-attr]
        low=0,
        high=samples.shape[0],
        size=(n_samples,)
    )
    random_samples = samples[random_integers, :]
    predictive_maps = model(random_samples)
    
    plt.figure(figsize=(4,9))  
    for i in range(n_samples):
        hp.projview(
            predictive_maps[i, :],
            sub=(n_samples, 1, i+1), # type: ignore
            cbar=False,
            nest=True,
            override_plot_properties={
                'figure_width': 3
            },
            **projview_kwargs
        )

def sky_probability(
    X: NDArray,
    lonlat: bool = False,
    nside: int = 256,
    smooth: None | float = 0.05,
    contour_levels: Sequence[float] | None = None,
    xsize: int = 500,
    rasterize_pmesh: bool = False,
    save_path: None | str = None,
    disable_mesh: bool = False,
    show: bool = True,
    no_axes: bool = False,
    truth_star: list | None = None,
    weights: NDArray | None = None,
    color: str | None = None,
    top_quad: bool = False,
    top_quad_mode: Literal["none", "legacy", "modern"] | None = None,
    **kwargs
) -> NDArray:
    '''
    :param nside: the resolution of the healpy map into which the samples
        are binned
    :param smooth: the sigma of the Gaussian kernel used to smooth the
        healpy sample map using healpy's smoothing function
    :param xsize: this specifies the resolution of the projection of the
        healpy map into matplotlib coords
    :param contour_levels: the significance levels (in units of sigma)
        defining how the contours are drawn (defaults to [1, 2] if None)
    :param colours: specify either a list of colours for each direction,
        or 'auto' to automatically assign colours
    :param rasterize_pmesh: specify whether or not to rasterize the
        pcolormesh, which tends to greatly increase the file size if
        not rasterized for high xsize
    :param save_path: if None, don't save; otherwise, save to path
    :param disable_mesh: supress plotting of the pcolormesh
    :param show: whether or not to call plt.show()
    :param no_axes: if false, do not create a blank set of Mollweide axes
    :param truth_star: add star indicating direction of true dipole, accepts
        list [phi, theta] in longtiude and colatitude in radians
    :param only_nth_direction: for a model with n directions, choose to
        plot only the nth directional posterior
    :param top_quad: crop saved output to the top-right quadrant of the Mollweide axes
    :param **kwargs: kwargs to be passed to hp.projview blank axes
    '''
    # Resolve requested quadrant behaviour.
    if top_quad_mode is None:
        chosen_mode: Literal["none", "legacy", "modern"]
        chosen_mode = "modern" if top_quad else "none"
    else:
        chosen_mode = top_quad_mode
    if chosen_mode not in {"none", "legacy", "modern"}:
        raise ValueError(f"Unsupported top_quadrant mode '{chosen_mode}'.")

    phi = X[:, -2]
    theta = X[:, -1]

    pdens_map = samples_to_hpmap(
        phi,
        theta,
        lonlat=lonlat,
        weights=weights,
        nside=nside,
        smooth=smooth,
    )

    selected_color = color or SKY_PROBABILITY_COLOR_CYCLE[0]
    c_chosen = matplotlib.colors.to_rgba(selected_color, alpha=0.4)
    c_white = matplotlib.colors.colorConverter.to_rgba('white', alpha=0)
    c_white_contourf = matplotlib.colors.colorConverter.to_rgba('white', alpha=0)
    cmap_cont = matplotlib.colors.LinearSegmentedColormap.from_list(
        'rb_cmap', [c_white_contourf, c_chosen], 512
    )
    cmaps = matplotlib.colors.LinearSegmentedColormap.from_list(
        'rb_cmap', [c_white, c_chosen], 512
    )

    use_modern = chosen_mode == "modern"
    ax = plt.gca()
    ax.set_rasterization_zorder(0)

    if use_modern:
        base_ax = ax
        proj_map = hp.mollview(
            pdens_map,
            return_projected_map=True,
            xsize=xsize,
            title="",
            cbar=False,
        )
        plt.close()

        proj_map = np.asarray(proj_map, dtype=np.float64)
        proj_map = np.nan_to_num(proj_map, nan=0.0, neginf=0.0, posinf=0.0)
        proj_map = np.clip(proj_map, 0.0, None)

        xx = np.linspace(-np.pi, np.pi, proj_map.shape[1])
        yy = np.linspace(-np.pi / 2.0, np.pi / 2.0, proj_map.shape[0])
        x_mask = xx >= 0.0
        y_mask = yy >= 0.0
        xx_quad = xx[x_mask]
        yy_quad = yy[y_mask]
        proj_map_quad = proj_map[np.ix_(y_mask, x_mask)]

        quad_sum = proj_map_quad.sum()
        if quad_sum > 0 and np.isfinite(quad_sum):
            proj_P_quad = proj_map_quad / quad_sum
        else:
            proj_P_quad = proj_map_quad

        Xa, Ya = np.meshgrid(xx_quad, yy_quad)
        levels = list(contour_levels) if contour_levels is not None else [1.0, 2.0]
        t_contours, _, _ = compute_2D_contours(proj_P_quad, levels)
        ax = base_ax

        if not no_axes:
            ax.cla()
            ax.set_facecolor("white")
            ax.set_rasterization_zorder(0)
        if not disable_mesh:
            im = ax.imshow(
                proj_P_quad,
                origin="lower",
                extent=[xx_quad.min(), xx_quad.max(), yy_quad.min(), yy_quad.max()],
                cmap=cmaps,
                rasterized=rasterize_pmesh,
                zorder=0,
            )
        contourf = ax.contourf(
            Xa,
            Ya,
            proj_P_quad,
            levels=t_contours,
            cmap=cmap_cont,
            zorder=1,
            extend="both",
        )
        contour = ax.contour(
            Xa,
            Ya,
            proj_P_quad,
            levels=t_contours,
            colors=[selected_color],
            zorder=1,
            extend="both",
        )
    else:
        if not no_axes:
            projview_kwargs = {
                'longitude_grid_spacing': 30,
                'color': 'white',
                'graticule': True,
                'graticule_labels': True,
                'cbar': False,
                **kwargs,
            }
            hp.projview(
                np.zeros(12),
                **projview_kwargs,
            )

        X_grid, Y_grid, proj_map = hp.projview(
            pdens_map, return_only_data=True, xsize=xsize
        )
        proj_map[proj_map == -np.inf] = 0.0
        proj_P_map = np.copy(proj_map)
        proj_P_map /= np.sum(proj_P_map)
        levels = list(contour_levels) if contour_levels is not None else [1.0, 2.0]
        t_contours, _, _ = compute_2D_contours(proj_P_map, levels)

        Xa, Ya = np.meshgrid(X_grid, Y_grid)
        if not disable_mesh:
            plt.pcolormesh(
                Xa,
                Ya,
                proj_P_map,
                cmap=cmaps,
                rasterized=rasterize_pmesh,
            )
        contourf = plt.contourf(
            Xa, Ya, proj_P_map, levels=t_contours, cmap=cmap_cont,
            zorder=1, extend='both'
        )
        contour = plt.contour(
            Xa, Ya, proj_P_map, levels=t_contours,
            colors=[selected_color], zorder=1, extend='both'
        )

    scatter_artists: list[Artist] = []
    if truth_star is not None:
        if isinstance(truth_star[0], (list, tuple, np.ndarray)):
            truth_iter = truth_star  # type: ignore[assignment]
        else:
            truth_iter = [truth_star]  # type: ignore[list-item]

        for idx, star in enumerate(truth_iter):
            phi_star = star[0]
            lat_star = star[1]
            if use_modern:
                lon_deg = np.rad2deg(phi_star)
                lat_deg = np.rad2deg(lat_star)
                x_proj, y_proj = _project_lonlat(lon_deg, lat_deg, wrap=False)
                x_plot = float(x_proj[0])
                y_plot = float(y_proj[0])
            else:
                x_plot = float(convert_to_l_dash(phi_star))
                y_plot = float(lat_star)
            scatter = plt.scatter(
                x_plot,
                y_plot,
                marker=marker_cycle[idx % len(marker_cycle)],
                color='black',
                s=100,
                zorder=20
            )
            scatter_artists.append(scatter)

    bbox_inches: str | Bbox = 'tight'

    if chosen_mode == "legacy" and save_path is not None:
        fig = plt.gcf()
        quad_bbox = get_top_quadrant_bbox(ax, fig)
        x_labels, y_labels = quad_tick_labels()
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.yaxis.tick_right()
        bbox_inches = quad_bbox

    elif use_modern:
        ax.set_xlim(0.0, np.pi)
        ax.set_ylim(0.0, np.pi / 2.0)
        patch = _ensure_top_quadrant_setup(ax)
        _clip_artists_to_patch(ax, patch)
        if not disable_mesh and 'im' in locals():
            im.set_clip_path(patch)
        for coll in getattr(contourf, "collections", []):
            coll.set_clip_path(patch)
        for coll in getattr(contour, "collections", []):
            coll.set_clip_path(patch)
        for scatter in scatter_artists:
            scatter.set_clip_path(patch)

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches=bbox_inches,
            pad_inches=0.0,
        )
    if show:
        plt.show()

    return proj_map

def smooth_map(
        healpy_map: NDArray,
        weights: NDArray | None = None,
        angle_scale: float = 1.,
        only_return_data: bool = False,
        fig: matplotlib.figure.Figure | None = None,
        **kwargs
    ) -> NDArray | None:
    smoothed_map_to_plot = average_smooth_map(
        healpy_map,
        weights=weights,
        angle_scale=angle_scale
    )

    if only_return_data:
        return smoothed_map_to_plot

    hp.projview(
        smoothed_map_to_plot,
        nest=True,
        fig=fig.number if fig is not None else None,
        **kwargs
    )
    return None
