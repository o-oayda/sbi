import numpy as np
import healpy as hp
from scipy.stats import poisson
import torch

def simulation(Theta):
    nside = 64
    Nbar = 50.0
    D, phi, theta = Theta

    pixel_indices = np.arange(hp.nside2npix(nside))
    pixel_vectors = hp.pix2vec(nside, pixel_indices)
    dipole_vector = D * hp.ang2vec(theta, phi)
    poisson_mean = Nbar * (1 + np.dot(dipole_vector, pixel_vectors))
    # dmap = poisson.rvs(mu=poisson_mean)
    print(poisson_mean)
    return poisson_mean

class PolarPrior:
    def __init__(self,
            theta_low, theta_high, return_numpy: bool = False
        ) -> None:
        self.return_numpy = return_numpy
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.ndim = len(theta_low)

    def sample(self, sample_shape=torch.Size([])):
        '''
        :param sample_shape: shape of output in batchwise format (Nsamp, Ndim)
        '''
        samples = self.generate_polar(sample_shape)
        return samples.numpy() if self.return_numpy else samples

    def generate_polar(self, shape: tuple):
        '''
        :param shape: shape of output in batchwise format (Nsamp, Ndim)
        :returns: uniform deviate on latitudinal (polar) angles

        - theta ~ acos(u * (cos theta_high - cos theta_low) + cos theta_low)
        - theta in [theta_low, theta_high], subdomain of [0, pi]
        '''
        # assert shape[-1] == 1
        u = torch.rand(shape)
        unif_theta = torch.arccos(
            torch.cos(self.theta_low)
            + u * (torch.cos(self.theta_high) - torch.cos(self.theta_low))
        )
        if len(shape) == 1:
            return unif_theta[:, None]
        else:
            return unif_theta

    def polar_logpdf(self, theta):
        '''Probably density of polar angle evaluated at theta for polar angles
        between [theta_low, theta_high]'''
        p_theta = - torch.sin(theta) / (
            torch.cos(self.theta_high) - torch.cos(self.theta_low)
        )
        return torch.log(p_theta)

    def log_prob(self, values):
        if self.return_numpy:
            values = torch.as_tensor(values)
        log_probs = self.polar_logpdf(values)
        return log_probs.numpy() if self.return_numpy else log_probs