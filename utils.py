import healpy as hp
import numpy as np
import torch
from torch import poisson

def sph2cart(theta_phi, device='cpu'):
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
    dipole_vector = D * sph2cart((theta, phi), device=device)
    poisson_mean = Nbar * (1 + torch.einsum('i,i...', dipole_vector, pixel_vectors))
    return poisson_mean

def simulation(Theta, nside=32, device='cpu'):
    poisson_mean = dipole_signal(Theta, nside, device)
    return poisson(poisson_mean)