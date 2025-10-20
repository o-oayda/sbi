from numpy.typing import NDArray
from dipolesbi.tools.utils import (
    compute_2D_contours, convert_to_l_dash, samples_to_hpmap
)
from dipolesbi.tools.maps import average_smooth_map
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
from typing import Callable, Sequence


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


def get_top_quadrant_bbox(ax: matplotlib.axes.Axes, fig: matplotlib.figure.Figure) -> Bbox:
    """Calculate the bounding box for the top-right quadrant of a Mollweide projection."""
    x0, x1 = -0.25, np.pi + 0.45
    y0, y1 = -0.22, np.pi / 2
    bbox = Bbox([[x0, y0], [x1, y1]])
    return bbox.transformed(ax.transData).transformed(fig.dpi_scale_trans.inverted())


try:  # optional torch dependency
    import torch
    from torch.types import Tensor
except ModuleNotFoundError:  # pragma: no cover - torch optional
    torch = None  # type: ignore
    Tensor = NDArray[np.float64]  # type: ignore

def posterior_predictive_check(
        samples: Tensor,
        model: Callable,
        n_samples: int = 5,
        **projview_kwargs
) -> None:
    if torch is None:
        raise RuntimeError("Torch is required for posterior_predictive_check but is not installed.")

    random_integers = torch.randint(  # type: ignore[union-attr]
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
    
    # plt.show()

def sky_probability(
    X: Tensor,
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
    **kwargs
) -> Tensor:
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
    # convert samples into desired coordinate system
    phi = X[:, -2]
    theta = X[:, -1]

    if not no_axes:
        hp.projview(
            np.zeros(12),
            **{
                'longitude_grid_spacing': 30,
                'color': 'white',
                'graticule': True,
                'graticule_labels': True,
                'cbar': False,
                **kwargs
            }
        )

    pdens_map = samples_to_hpmap(
        phi,
        theta,
        lonlat=lonlat,
        weights=weights,
        nside=nside,
        smooth=smooth,
    )

    X, Y, proj_map = hp.projview(
        pdens_map, return_only_data=True, xsize=xsize
    )

    # the proj_map will have more bins therefore will not sum to one
    # it also has -infs; remove these and renormalise
    proj_map[proj_map == -np.inf] = 0
    proj_P_map = np.copy(proj_map)
    proj_P_map /= np.sum(proj_P_map)
    levels = list(contour_levels) if contour_levels is not None else [1.0, 2.0]
    t_contours, _, _ = compute_2D_contours(proj_P_map, levels)

    selected_color = color or SKY_PROBABILITY_COLOR_CYCLE[0]
    c_chosen = matplotlib.colors.to_rgba(selected_color, alpha=0.4)
    c_white = matplotlib.colors.colorConverter.to_rgba(
        'white',alpha=0
    )
    c_white_contourf = matplotlib.colors.colorConverter.to_rgba(
        'white',alpha=0
    )
    cmap_cont = matplotlib.colors.LinearSegmentedColormap.from_list(
                'rb_cmap',[c_white_contourf, c_chosen], 512
    )
    cmaps = matplotlib.colors.LinearSegmentedColormap.from_list(
        'rb_cmap',[c_white, c_chosen], 512
    )

    Xa, Ya = np.meshgrid(X, Y)
    ax = plt.gca()
    ax.set_rasterization_zorder(0)

    if not disable_mesh:
        plt.pcolormesh(
            Xa, Ya, proj_P_map, cmap=cmaps,
            rasterized=rasterize_pmesh,
        )

    plt.contourf(
        Xa, Ya, proj_P_map, levels=t_contours, cmap=cmap_cont,
        zorder=1, extend='both'
    )
    plt.contour(
        Xa, Ya, proj_P_map, levels=t_contours,
        colors=[selected_color], zorder=1, extend='both'
    )
    if truth_star is not None:
        if isinstance(truth_star[0], (list, tuple, np.ndarray)):
            truth_iter = truth_star  # type: ignore[assignment]
        else:
            truth_iter = [truth_star]  # type: ignore[list-item]

        for idx, star in enumerate(truth_iter):
            phi_star = star[0]
            lat_star = star[1]
            plt.scatter(
                convert_to_l_dash(phi_star),
                lat_star,
                marker=marker_cycle[idx % len(marker_cycle)],
                color='black',
                s=100,
                zorder=20
            )

    # pcolormesh already rasterized via kwarg.

    bbox_inches: str | Bbox = 'tight'
    if save_path is not None and top_quad:
        fig = plt.gcf()
        quad_bbox = get_top_quadrant_bbox(ax, fig)
        x_labels, y_labels = quad_tick_labels()
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.yaxis.tick_right()
        bbox_inches = quad_bbox

    if save_path is not None:
        plt.savefig(
            save_path, dpi=300, bbox_inches=bbox_inches
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
