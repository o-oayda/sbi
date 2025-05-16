from dipolesbi.tools.utils import (
    compute_2D_contours, convert_to_l_dash, samples_to_hpmap
)
from dipolesbi.tools.maps import average_smooth_map
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.types import Tensor

def sky_probability(
    X: Tensor,
    lonlat: bool = False,
    nside: int = 256,
    smooth: None | float = 0.05,
    contour_levels: list[float] = [0.5, 1, 1.5, 2],
    xsize: int = 500,
    rasterize_pmesh: bool = False,
    save_path: None | str = None,
    disable_mesh: bool = False,
    show: bool = True,
    no_axes: bool = False,
    truth_star: list | None = None,
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

    pdens_map = samples_to_hpmap(phi, theta,
        nside=nside, smooth=smooth, lonlat=lonlat
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

    c_chosen = matplotlib.colors.to_rgba('tomato', alpha=0.4)
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
    plt.contourf(
        Xa, Ya, proj_P_map, levels=t_contours, cmap=cmap_cont,
        zorder=1, extend='both'
    )
    plt.contour(
        Xa, Ya, proj_P_map, levels=t_contours,
        colors=['tomato'], zorder=1, extend='both'
    )
    if truth_star is not None:
        phi_star, theta_star = truth_star[0], np.pi/2 - truth_star[1]

        plt.scatter(
            convert_to_l_dash(phi_star),
            theta_star,
            marker='*',
            color='black',
            s=150,
            zorder=20
        )

    if not disable_mesh:
        plt.pcolormesh(
            Xa, Ya, proj_P_map, cmap=cmaps,
            rasterized=rasterize_pmesh,
        )

    if save_path is not None:
        plt.savefig(
            save_path, dpi=300, bbox_inches='tight'
        )
    if show:
        plt.show()

    return proj_map

def smooth_map(
        healpy_map: Tensor,
        weights: Tensor | None = None,
        angle_scale: float = 1.,
        **kwargs
    ) -> None:
    smoothed_map_to_plot = average_smooth_map(
        healpy_map,
        weights=weights,
        angle_scale=angle_scale
    )
    hp.projview(
        smoothed_map_to_plot.detach().cpu().numpy(),
        nest=True,
        **kwargs
    )