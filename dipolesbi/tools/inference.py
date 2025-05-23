from dipolesbi.tools.utils import save_simulation
import dynesty
import emcee
import numpy as np
import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import hpCNNEmbedding
from sbi.utils.user_input_checks import process_prior
import pickle
import healpy as hp
import os
from numpy.typing import NDArray

class Inference:
    def __init__(self, prior=None, model=None):
        self.prior = prior
        self.model = model

    def run_mcmc(self,
        nwalkers: int = 32,
        n_steps: int = 2000,
        burn_in: int = 100
    ) -> None:
        '''
        Run MCMC with emcee and save posterior in samples attribute.
        '''

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
        '''
        Begin the nested sampling process and save results in dresults attribute.
        '''
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

    def make_batch_simulations(self,
            n_simulations: int = 2000,
            n_workers: int = 32,
            device: str = 'cpu',
            save: bool = False,
            custom_save_dir: str | None = None
    ) -> None:
        self.prior.to(device)
        self.prior, num_parameters, prior_returns_numpy = process_prior(
            self.prior,
            custom_prior_wrapper_kwargs={
                'lower_bound': torch.as_tensor(
                    self.prior.get_low_ranges(), device=self.prior.device
                ),
                'upper_bound': torch.as_tensor(
                    self.prior.get_high_ranges(), device=self.prior.device
                )
            }
        )
        self.theta, self.x = self.model.batch_simulator(
            self.prior,
            n_samples=n_simulations,
            n_workers=n_workers,
            prior_returns_numpy=prior_returns_numpy,
        )

        if save:
            save_simulation(
                theta=self.theta,
                x=self.x,
                prior=self.prior,
                custom_save_dir=custom_save_dir
            )
        
        return (self.theta, self.x)
            
    def run_sbi(self,
            sim_dir: str | None,
            device: str = 'cpu',
    ) -> None:
        if sim_dir is not None:
            self.load_simulation(sim_dir)

        # do the training on the gpu but not the simulation
        self.theta = self.theta.to(device); self.x = self.x.to(device)

        # choose which type of pre-configured embedding net to use (e.g. CNN)
        # must be nested healpix ordering!!!
        if not hasattr(self, 'nside'):
            self.nside = hp.npix2nside(self.x.shape[-1])
        embedding_net = hpCNNEmbedding(nside=self.nside)

        # instantiate the conditional neural density estimator
        # maf, maf_rqs 
        neural_posterior = posterior_nn(
            model="maf",
            embedding_net=embedding_net
        )
        inference = NPE(
            prior=self.prior,
            density_estimator=neural_posterior,
            device=device
        )

        inference = inference.append_simulations(self.theta, self.x)
        density_estimator = inference.train(show_train_summary=True)
        self.posterior = inference.build_posterior(
            density_estimator,
            prior=self.prior,
        )
        print(self.posterior)
    
    def load_simulation(self, sim_dir: str) -> None:
        if not os.path.exists(f'simulations/{sim_dir}/'):
            raise FileNotFoundError(f'Cannot find {sim_dir}.')
        
        print(f'Opening {sim_dir}...')
        sim_path = f'simulations/{sim_dir}/theta_and_x.pt'
        self.theta, self.x = torch.load(sim_path)
        
        prior_path = f'simulations/{sim_dir}/prior.pkl'
        print(f'Opening {prior_path}...')
        with open(prior_path, "rb") as handle:
            self.prior = pickle.load(handle)
    
    def save_posterior(self, file_path: str) -> None:
        print(f'Saving to {file_path}...')
        with open(file_path, "wb") as handle:
            pickle.dump(self.posterior, handle)
    
    def load_posterior(self, file_path: str) -> None:
        print(f'Opening {file_path}...')
        with open(file_path, "rb") as handle:
            self.posterior = pickle.load(handle)

    def sample_amortized_posterior(self,
            x_obs,
            n_samps: int = 10_000
    ) -> NDArray[np.float64]:
        return self.posterior.sample((n_samps,), x=x_obs).cpu().detach().numpy()