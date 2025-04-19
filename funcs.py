import numpy as np
import healpy as hp
from torch import poisson
from scipy.stats import poisson as sp_poisson
import torch
import emcee
import dynesty
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import hpCNNEmbedding
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.inference import NPE, simulate_for_sbi
from priors import DipolePrior
from torch.types import Tensor

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

def simulation(Theta, nside=32, device='cpu'):
    poisson_mean = dipole_signal(Theta, nside, device)
    return poisson(poisson_mean)

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

class SkyMap:
    def __init__(self, nside: int = 32, device: str = 'cpu'):
        self.nside = nside
        self.device = device
        self.mask = Mask(self.nside)
        self.mask_map = torch.zeros(self.mask.npix)
    
    def generate_dipole(self, Theta: Tensor) -> None:
        poisson_mean = self.dipole_signal(Theta)
        self._density_map = poisson(poisson_mean)

    # def dipole_signal(self, Theta) -> Tensor:
    #     Nbar, D, phi, theta = torch.as_tensor(
    #         Theta, device=self.device, dtype=torch.float64
    #     )
    #     pixel_indices = torch.arange(hp.nside2npix(self.nside))
    #     pixel_vectors = torch.as_tensor(
    #         torch.stack(
    #             hp.pix2vec(self.nside, pixel_indices, nest=True)
    #         ),
    #         device=self.device
    #     )
    #     dipole_vector = D * sph2cart((theta, phi), device=self.device)
    #     poisson_mean = Nbar * (1 + torch.einsum('i,i...', dipole_vector, pixel_vectors))
    #     return poisson_mean
    
    def dipole_signal(self, Theta: Tensor) -> Tensor:
        if Theta.shape == (4,):
            Theta = Theta.reshape(1,4)
        
        n_batches = Theta.shape[0]
        pixel_indices = torch.arange(hp.nside2npix(self.nside))
        pixel_vectors = torch.as_tensor(
            torch.stack(
                hp.pix2vec(self.nside, pixel_indices, nest=True)
            ),
            device=self.device
        ).to(torch.float32)
        mean_count = Theta[:, 0]
        dipole_amplitude = Theta[:, 1]
        dipole_longitude = Theta[:, 2]
        dipole_latitude = Theta[:, 3]
        dipole_vector = dipole_amplitude.reshape((n_batches,1)) * sph2cart(
            (dipole_latitude, dipole_longitude),
            device=self.device
        )
        poisson_mean = mean_count.reshape((n_batches,1)) * (
            1 + torch.einsum('ij,jk', dipole_vector, pixel_vectors)
        )
        # poisson_mean = mean_count * (
            # 1 + torch.einsum('ik,ji->jk', dipole_vector, pixel_vectors.T)
        # )
        if n_batches == 1:
            return poisson_mean.flatten()
        else:
            return poisson_mean

    def mask_pixels(self, fill_value = None, **kwargs) -> None:
        self.kwarg_to_mask = {'equator_mask': self.mask.equator_mask}
        for key, val in kwargs.items():
            masked_pixel_indices = self.kwarg_to_mask[key](val)
            self.mask_map[masked_pixel_indices] = 1
        
        if fill_value == None:
            self.fill_value = torch.nan
        else:
            self.fill_value = fill_value

    @property
    def density_map(self):
        out = self._density_map
        out[self.mask_map == 1] = self.fill_value
        return out
    
    def batch_simulator(self,
            proposal_distribution,
            n_samples: int,
            mask_fill_value = None,
            **mask_kwargs
    ) -> tuple[Tensor]:
        Theta = proposal_distribution.sample((n_samples,))
        poisson_mean = self.dipole_signal(Theta)
        self.batch_density_maps = poisson(poisson_mean)

        if mask_fill_value == None:
            fill_value = torch.nan
        else:
            fill_value = mask_fill_value
        
        self.mask_pixels(**mask_kwargs)
        self.batch_mask_maps = self.mask_map.repeat((n_samples, 1))
        self.batch_mask_maps[self.batch_mask_maps == 1] = fill_value

        return (Theta, self.batch_density_maps)

class Mask:
    def __init__(self, nside: int = 32):
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        self.all_pixel_indices = set(np.arange(self.npix))
    
    def equator_mask(self, mask_angle: float) -> list:
        south_pole_vec = hp.ang2vec(0, -90, lonlat=True)
        north_pole_vec = hp.ang2vec(0, 90, lonlat=True)
        north_pole_indices = hp.query_disc(
            self.nside, north_pole_vec, radius=np.deg2rad(90 - mask_angle)
        )
        south_pole_indices = hp.query_disc(
            self.nside, south_pole_vec, radius=np.deg2rad(90 - mask_angle)
        )
        masked_pixel_indices = (
            self.all_pixel_indices
            - set([*north_pole_indices, *south_pole_indices])
        )
        return list(masked_pixel_indices)

class Inference:
    def __init__(self):
        pass
    
    def run_mcmc(self, nwalkers=32, n_steps=2000, burn_in=100):
        
        def log_prob(Theta):
            log_prior = self.log_prior_likelihood(Theta)
            if not np.isfinite(log_prior):
                return -np.inf
            return log_prior + self.log_likelihood(Theta)        
        
        pos = np.zeros((nwalkers, self.ndim))
        for i in range(0, nwalkers):
            unifs = np.random.rand(self.ndim)
            pos[i, :] = self.prior_transform(unifs)
        
        self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, log_prob)
        self.sampler.run_mcmc(pos, n_steps, progress=True)
        self.samples = self.sampler.get_chain(discard=burn_in, flat=True)

    def run_dynesty(self,
        sample_method: str = 'auto',
        print_info: bool = True,
        **kwargs
    ):
        '''Begin the nested sampling process and return the results upon
        completion.'''
        dsampler = dynesty.NestedSampler(
            self.log_likelihood,
            self.prior_transform,
            **{
                'ndim': self.ndim,
               'sample': sample_method,
               **kwargs
            }
        )
        
        dsampler.run_nested(print_progress=print_info)
        self.model_evidence = dsampler.results.logz[-1]
        print('Model evidence: {:.2f}'.format(self.model_evidence))
        self.dresults = dsampler.results
    
    def run_sbi(self,
            n_simulations: int = 2000,
            device: str = 'cpu'
    ) -> None:
        prior = DipolePrior(
            mean_count_range=self.mean_count_range,
            amplitude_range=self.amplitude_range,
            longitude_range=self.longitude_range,
            latitude_range=self.latitude_range
        )
        self.theta, self.x = SkyMap().batch_simulator(
            prior, n_samples=n_simulations
        )
        
        # do the training on the gpu but not the simulation
        self.theta = self.theta.to(device); self.x = self.x.to(device)
        prior.to(device)
        
        prior, num_parameters, prior_returns_numpy = process_prior(
            prior,
            custom_prior_wrapper_kwargs={
                'lower_bound': torch.as_tensor(
                    prior.get_low_ranges(), device=prior.device
                ),
                'upper_bound': torch.as_tensor(
                    prior.get_high_ranges(), device=prior.device
                )
            }
        )
        # choose which type of pre-configured embedding net to use (e.g. CNN)
        # must be nested healpix ordering!!!
        embedding_net = hpCNNEmbedding(nside=self.nside) 

        # instantiate the conditional neural density estimator
        # maf, maf_rqs 
        neural_posterior = posterior_nn(
            model="maf", embedding_net=embedding_net
        )
        inference = NPE(
            prior=prior, density_estimator=neural_posterior, device=device
        )

        inference = inference.append_simulations(self.theta, self.x)
        density_estimator = inference.train(show_train_summary=True)
        self.posterior = inference.build_posterior(
            density_estimator, prior=prior,
        )
        print(self.posterior)
    
    def sample_amortized_posterior(self, x_obs, n_samps: int = 10_000):
        return self.posterior.sample((n_samps,), x=x_obs).cpu().detach().numpy()

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