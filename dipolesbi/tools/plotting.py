from numpy.typing import NDArray
from dipolesbi.tools.utils import (
    compute_2D_contours, convert_to_l_dash, samples_to_hpmap
)
from dipolesbi.tools.maps import average_smooth_map
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable


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
    contour_levels: list[float] = [1., 2.],
    xsize: int = 500,
    rasterize_pmesh: bool = False,
    save_path: None | str = None,
    disable_mesh: bool = False,
    show: bool = True,
    no_axes: bool = False,
    truth_star: list | None = None,
    weights: NDArray | None = None,
    color: str | None = None,
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
        defining how the contours are drawm
    :param colours: specify either a list of colours for each direction,
        or 'auto' to automatically assign colours
    :param rasterize_pmesh: specify whether or not to rasterize the
        pcolormesh, which tends to greatly increase the file size if
        not rasterized for high xsize
    :param save_path: if None, don't save; otherwise, save to path
    :param disable_mesh: supress plotting of the pcolormesh
    :param show: whether or not to call plt.show()
    :param no_axes: if false, do not create a blank set of Mollweide axes
    :param top_quad: if save_path has been specified, specifying top_quad
        to true creates a bounding box which restricts the saved figure
        to the top right quadrant of the Mollweide Axes
    :param truth_star: add star indicating direction of true dipole, accepts
        list [phi, theta] in longtiude and colatitude in radians
    :param only_nth_direction: for a model with n directions, choose to
        plot only the nth directional posterior
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
    t_contours, _, _ = compute_2D_contours(proj_P_map, contour_levels)

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

    if save_path is not None:
        plt.savefig(
            save_path, dpi=300, bbox_inches='tight'
        )
    if show:
        plt.show()

    return proj_map

def smooth_map(
        healpy_map: NDArray,
        weights: NDArray | None = None,
        angle_scale: float = 1.,
        only_return_data: bool = False,
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
        **kwargs
    )
