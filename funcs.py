import numpy as np
import healpy as hp
from scipy.stats import poisson
import torch
import emcee
import dynesty
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.utils import BoxUniform
from sbi.inference import NPE, simulate_for_sbi

def simulation(Theta, nside=64):
    poisson_mean = dipole_signal(Theta, nside)
    return poisson.rvs(mu=poisson_mean)

def dipole_signal(Theta, nside=64):
    Nbar, D, phi, theta = Theta
    pixel_indices = np.arange(hp.nside2npix(nside))
    pixel_vectors = hp.pix2vec(nside, pixel_indices)
    dipole_vector = D * hp.ang2vec(theta, phi)
    poisson_mean = Nbar * (1 + np.dot(dipole_vector, pixel_vectors))
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
    
    def run_sbi(self):
        prior_nbar = BoxUniform(
            low=0*torch.ones(1), high=100*torch.ones(1)
        )
        prior_d = BoxUniform(
            low=0*torch.ones(1), high=0.1*torch.ones(1)
        )
        prior_phi = BoxUniform(
            low=0*torch.ones(1), high=2*torch.pi*torch.ones(1)
        )
        prior_theta = PolarPrior(
            theta_low=0*torch.ones(1), theta_high=torch.pi*torch.ones(1)
        )
        prior, num_parameters, prior_returns_numpy = process_prior(
            [prior_nbar, prior_d, prior_phi, prior_theta],
        )
        
        simulator = process_simulator(simulation, prior, prior_returns_numpy)
        check_sbi_inputs(simulator, prior)
        inference = NPE(prior=prior)

        n_workers = 32
        n_simulations = 2000
        theta, x = simulate_for_sbi(
            simulator,
            proposal=prior, num_simulations=n_simulations, num_workers=n_workers,
            show_progress_bar=True
        )

        inference = inference.append_simulations(theta, x)
        density_estimator = inference.train()
        posterior = inference.build_posterior(density_estimator)
        print(posterior)

        self.samples = posterior.sample(
            (10000,), x=self.density_map
        ).cpu().detach().numpy()

class DipolePoisson(Inference):
    def __init__(self, density_map, amplitude_high=0.1, mean_count_high=100):
        super().__init__()
        self.amplitude_high = amplitude_high
        self.mean_count_high = mean_count_high
        self.density_map = density_map
        self.ndim = 4
    
    def prior_transform(self, uTheta):
        u_mean_count, u_amplitude, u_longitude, u_colatitude = uTheta
        mean_count = self.mean_count_high * u_mean_count
        amplitude = self.amplitude_high * u_amplitude
        longitude = 2 * np.pi * u_longitude
        latitude = np.arccos(1 - 2 * u_colatitude)
        return mean_count, amplitude, longitude, latitude

    def log_prior_likelihood(self, Theta):
        mean_count, amplitude, longitude, latitude = Theta
        conditions = [
            0 < mean_count < self.mean_count_high,
            0 < amplitude < self.amplitude_high,
            0 < longitude < 2 * np.pi,
            0 < latitude < np.pi
        ]
        if not all(conditions):
            return -np.inf
        else:        
            P_mean_count = np.log( 1 / self.mean_count_high )
            P_amplitude = np.log( 1 / self.amplitude_high )
            P_longitude = np.log( 1 / (2 * np.pi) )
            P_colatitude = np.log( 0.5 * np.sin(latitude) )
            return P_mean_count + P_amplitude + P_longitude + P_colatitude
    
    def log_likelihood(self, Theta):
        signal = dipole_signal(Theta)
        return np.sum(
            poisson.logpmf(k=self.density_map, mu=signal)
        )