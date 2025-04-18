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
    return torch.as_tensor((x, y, z), device=device)

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
            posterior_device: str = 'cpu'
    ) -> None:
        prior = DipolePrior(
            mean_count_range=self.mean_count_range,
            amplitude_range=self.amplitude_range,
            longitude_range=self.longitude_range,
            latitude_range=self.latitude_range
        )
        prior.to(self.device)
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
        # do the training on the gpu but not the simulation
        # simulation_device = lambda x: simulation(x, device=self.device)
        simulator = process_simulator(simulation, prior, prior_returns_numpy)
        check_sbi_inputs(simulator, prior)

        # choose which type of pre-configured embedding net to use (e.g. CNN)
        # must be nested healpix ordering!!!
        embedding_net = hpCNNEmbedding(nside=self.nside) 

        # instantiate the conditional neural density estimator
        # maf, maf_rqs 
        neural_posterior = posterior_nn(
            model="maf", embedding_net=embedding_net
        )
        inference = NPE(
            prior=prior, density_estimator=neural_posterior, device=self.device
        )

        n_workers = 32
        self.theta, self.x = simulate_for_sbi(
            simulator,
            proposal=prior, num_simulations=n_simulations,
            num_workers=n_workers, show_progress_bar=True
        )

        inference = inference.append_simulations(self.theta, self.x)
        density_estimator = inference.train(show_train_summary=True)
        self.posterior = inference.build_posterior(
            density_estimator, prior=prior,
        )
        print(self.posterior)
        # self.posterior.to(device=posterior_device)
    
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
        signal = dipole_signal(Theta)
        return np.sum(
            sp_poisson.logpmf(k=self.density_map, mu=signal)
        )