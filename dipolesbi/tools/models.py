from dipolesbi.tools.maps import SkyMap
from dipolesbi.tools.inference import Inference
from dipolesbi.tools.utils import polar_pdf, sample_polar, sample_unif, unif_pdf
import healpy as hp
import numpy as np
import torch
from scipy.stats import poisson as sp_poisson

class DipolePoisson(Inference):
    def __init__(self,
            density_map: np.ndarray[float],
            amplitude_range: list[float] = [0, 0.1],
            mean_count_range: list[float] = [0,100.0],
            longitude_range: list[float] = [0, 2*np.pi],
            latitude_range: list[float] = [0, np.pi],
            device: str = 'cpu'
    ) -> None:
        super().__init__()
        self.density_map = density_map
        self.npix = len(density_map)
        self.nside = hp.npix2nside(self.npix)
        self.ndim = 4
        self.device = device
        self.sky_map = SkyMap(self.nside, device=self.device)
        self.mean_count_range = mean_count_range
        self.amplitude_range = amplitude_range
        self.longitude_range = longitude_range
        self.latitude_range = latitude_range

    def prior_transform(self, uTheta):
        u_mean_count, u_amplitude, u_longitude, u_colatitude = uTheta
        mean_count = sample_unif(u_mean_count, self.mean_count_range)
        amplitude = sample_unif(u_amplitude, self.amplitude_range)
        longitude = sample_unif(u_longitude, self.longitude_range)
        latitude = sample_polar(u_colatitude, self.latitude_range)
        return mean_count, amplitude, longitude, latitude

    def log_prior_likelihood(self, Theta):
        mean_count, amplitude, longitude, latitude = Theta
        conditions = [
            self.mean_count_range[0] < mean_count < self.mean_count_range[1],
            self.amplitude_range[0]  < amplitude  < self.amplitude_range[1],
            self.longitude_range[0]  < longitude  < self.longitude_range[1],
            self.latitude_range[0]   < latitude   < self.latitude_range[1]
        ]
        if not all(conditions):
            return -np.inf
        else:
            P_mean_count = np.log( unif_pdf (  self.mean_count_range ) )
            P_amplitude  = np.log( unif_pdf (  self.amplitude_range ) )
            P_longitude  = np.log( unif_pdf (  self.longitude_range ) )
            P_colatitude = np.log( polar_pdf(  latitude, self.latitude_range ) )
            return P_mean_count + P_amplitude + P_longitude + P_colatitude

    def log_likelihood(self, Theta):
        signal = self.sky_map.dipole_signal(
            torch.as_tensor(Theta).to(torch.float32)
        )
        return np.sum(
            sp_poisson.logpmf(k=self.density_map, mu=signal)
        )