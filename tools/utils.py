import healpy as hp
import numpy as np
import torch
from torch import poisson
from torch.types import Tensor
from scipy.interpolate import interp1d
import os
import pickle
from astropy.coordinates import SkyCoord
import astropy.units as u

def spherical_to_cartesian(theta_phi, device='cpu'):
    '''
    Transform spherical coordinates in the form (theta, phi) to Cartesian
    coordinates given r = 1. Theta is the polar angle and phi the azimuthal
    angle. The polar angle runs from 0 to 180 degrees, where zero degrees
    corresponds to z = 1 in Cartesian coordinates. From Honours code.
    '''
    x = torch.sin(theta_phi[0]) * torch.cos(theta_phi[1])
    y = torch.sin(theta_phi[0]) * torch.sin(theta_phi[1])
    z = torch.cos(theta_phi[0])
    xyz = torch.stack([x, y, z], dim=1)
    return xyz.to(device=device)

def sample_unif(unif: float, low_high: list[float]) -> float:
    '''
    (b - a) * u + a
    '''
    low = low_high[0]; high = low_high[1]
    return (high - low) * unif + low

def unif_pdf(low_high: list[float]) -> float:
    low = low_high[0]; high = low_high[1]
    return 1 / (high - low)

def sample_polar(unif: float, low_high: list[float]) -> float:
    low = low_high[0]; high = low_high[1]
    unif_theta = np.arccos(np.cos(low) + unif * (np.cos(high) - np.cos(low)))
    return unif_theta

def polar_pdf(theta: float, low_high: list[float]):
    low = low_high[0]; high = low_high[1]
    return - np.sin(theta) / (np.cos(high) - np.cos(low))

def dipole_signal(Theta, nside=32, device='cpu'):
    Nbar, D, phi, theta = torch.as_tensor(Theta, device=device, dtype=torch.float64)
    pixel_indices = torch.arange(hp.nside2npix(nside))
    pixel_vectors = torch.as_tensor(
        torch.stack(
            hp.pix2vec(nside, pixel_indices, nest=True)
        ),
        device=device
    )
    dipole_vector = D * spherical_to_cartesian((theta, phi), device=device)
    poisson_mean = Nbar * (1 + torch.einsum('i,i...', dipole_vector, pixel_vectors))
    return poisson_mean

def simulation(Theta, nside=32, device='cpu'):
    poisson_mean = dipole_signal(Theta, nside, device)
    return poisson(poisson_mean)

def check_vectorised_input(Theta: Tensor, ndim: int) -> Tensor:
        if Theta.shape == (ndim,):
            Theta = Theta.reshape(1, ndim)
        return Theta

def convert_to_l_dash(l):
    '''When plotting the dynesty histogram on top of the healpy projection
    plot, for whatever reason despite using galactic coordinates the healpy
    plot needs the values for l to be between [-pi and pi], not [0 and 2pi].
    This performs the conversion accordingly.
    
    Parameters
    ----------
    l : the azimuthal angle of each point in galactic coordinates in radians,
    moving from 0 to 2pi.'''

    try:
        l_dash = []
        for i in range(0,len(l)):
            if l[i] <= np.pi:
                l_dash.append(-l[i])
            elif l[i] > np.pi:
                l_dash.append(2 * np.pi - l[i])
            else:
                print('No you should not be here!')
        return np.array(l_dash)
    except TypeError:
        if l <= np.pi:
            return -l
        elif l > np.pi:
            return 2 * np.pi - l
        else:
            print('No you should not be here!')

def sigma_to_prob2D(sigma: list) -> Tensor:
    '''Convert sigma significance to mass enclosed inside a 2D normal
    distribution using the explicit formula for a 2D normal.
    :param sigma: the levels of significance
    :returns: the probability enclosed within each significance level'''
    return 1.0 - np.exp(-0.5 * np.asarray(sigma)**2)

def compute_2D_contours(
        P_xy: Tensor,
        contour_levels: list[float]
) -> tuple[Tensor]:
    '''
    Compute contour heights corresponding to sigma levels of probability
    density by creating a mapping (interpolation function) from the CDF
    (enclosed prob) to some arbitrary level of probability density.
    from here: https://stackoverflow.com/questions/37890550/python-plotting-percentile-contour-lines-of-a-probability-distribution

    :param P_xy: normalised 2D probability (not density)
    :param contour_levels: pass list of sigmas at which to draw the contours
    :return:
        1. vector of probabilities corresponding to heights at which to
        draw the contours (pass to e.g. plt.contour with levels= kwarg);
        2. uniformly spaced probability levels between 0 and max prob
        3. CDF at given P_level, of length 1000 (the hardcoded number of
            P_levels)
    '''
    P_levels = np.linspace(0, P_xy.max(), 1000)
    mask = (P_xy >= P_levels[:, None, None])
    P_integral = (mask * P_xy).sum(axis=(1,2))
    f = interp1d(P_integral, P_levels)
    t_contours = np.flip(f(sigma_to_prob2D(contour_levels)))
    return t_contours, P_levels, P_integral

def samples_to_hpmap(
        phi: Tensor,
        theta: Tensor,
        lonlat: bool = False,
        nside: int = 64,
        smooth: None | float = None
    ) -> Tensor:
    '''
    Turn numerical samples in phi-theta space to a healpy map, in the native
    coords of phi theta, defining the probability of a sample (phi_i, theta_i)
    lying in a given pixel.

    :param phi: vector of phi samples in spherical coordinates: [0, 2pi)
    :param theta: vector of theta samples in spherical coordinates: [0, pi]
    :param lonlat: if True, phi ~ [0, 360] and theta ~ [-90, 90]; else,
        phi ~ [0, 2pi] and theta ~ [0, pi]
    :param weights: weights of each samples, defaults to None
    :param nside: nside (resolution) of binning of nested samples, i.e. the
        resolution of the posterior probability map
    '''
    # note: labels flip where lonlat=True... thanks healpy
    if lonlat:
        sample_pixel_indices = hp.ang2pix(
            nside=nside, theta=phi, phi=theta, lonlat=lonlat
        )
    else:
        sample_pixel_indices = hp.ang2pix(
            nside=nside, theta=theta, phi=phi
        )
    sample_count_map = np.bincount(
        sample_pixel_indices, minlength=hp.nside2npix(nside)
    )
    
    # convert count to prob density
    map_total = np.sum(sample_count_map)
    sample_pdensity_map = sample_count_map / map_total
    
    if smooth is not None:
        # healpy's smooth function (in samples_to_hpmap) works in sph
        # harmonic space, and produces a small number of very small
        # negative values; the sum is also not preserved.
        # correct by replacing negative values with 0 and renormalise.
        smooth_map = hp.sphtfunc.smoothing(sample_pdensity_map, sigma=smooth)
        smooth_map[smooth_map < 0] = 0
        smooth_map /= np.sum(smooth_map)
        return smooth_map
    else:
        return sample_pdensity_map

def save_simulation(theta: Tensor, x: Tensor, prior) -> None:
    if not os.path.exists('simulations/'):
        os.makedirs('simulations/')
    i = 1
    while os.path.exists(f'simulations/sim{i}/'):
        i += 1
    base_path = f'simulations/sim{i}'
    os.makedirs(f'{base_path}/')
    simulation_path = f'{base_path}/theta_and_x.pt'
    prior_path = f'{base_path}/prior.pkl'

    print(f'Saving theta and x to {simulation_path}...')
    torch.save([theta, x], simulation_path)

    print(f'Saving prior to {prior_path}...')
    with open(prior_path, "wb") as handle:
        pickle.dump(prior, handle)

def load_simulation(sim_dir: str) -> tuple:
    if not os.path.exists(f'simulations/{sim_dir}/'):
        raise FileNotFoundError(f'Cannot find {sim_dir}.')
    
    print(f'Opening {sim_dir}...')
    sim_path = f'simulations/{sim_dir}/theta_and_x.pt'
    theta, x = torch.load(sim_path)
    
    prior_path = f'simulations/{sim_dir}/prior.pkl'
    print(f'Opening {prior_path}...')
    with open(prior_path, "rb") as handle:
        prior = pickle.load(handle)
    
    return theta, x, prior

def omega_to_theta(omega):
    '''
    Convert solid angle in steradins to theta in radians for
    a cone section of a sphere.
    
    :param omega: solid angle in steradians.
    '''
    return torch.arccos( 1 - omega / (2 * np.pi) )

def equatorial_to_ecliptic(ra, dec, output_unit='radians'):
    eq = SkyCoord(ra, dec, unit=u.deg)
    ecl = eq.transform_to('barycentricmeanecliptic')
    if output_unit == 'radians':
        return ecl.lon.rad, ecl.lat.rad
    elif output_unit == 'degrees':
        return ecl.lon.deg, ecl.lat.deg
    else:
        raise Exception(
            'Not a valid unit. Select either radians of degrees.')