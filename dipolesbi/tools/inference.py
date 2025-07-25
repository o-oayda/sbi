from dipolesbi.tools.utils import save_simulation
import dynesty
import emcee
import numpy as np
import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import hpCNNEmbedding
from sbi.utils.user_input_checks import process_prior
from sbi.inference import simulate_for_sbi
from sbi.utils.user_input_checks import process_simulator, check_sbi_inputs
from sbi.diagnostics import check_sbc, run_sbc, check_tarp, run_tarp
from sbi.analysis.plot import sbc_rank_plot
import pickle
import healpy as hp
import os
from numpy.typing import NDArray
from torch.types import Tensor
from typing import Optional
from dipolesbi.tools.utils import Samples
from dipolesbi.tools.plotting import smooth_map
import matplotlib.pyplot as plt

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
                    self.prior.low_ranges, device=self.prior.device
                ),
                'upper_bound': torch.as_tensor(
                    self.prior.high_ranges, device=self.prior.device
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
        self._check_for_mask_nans()

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
            n_samps: int = 10_000,
            **kwargs
    ) -> NDArray[np.float64]:
        return self.posterior.sample((n_samps,), x=x_obs, **kwargs).cpu().detach().numpy()
    
    def _check_for_mask_nans(self) -> None:
        assert hasattr(self, 'x'), 'Load the data first.'

        if torch.isnan(self.x).any():
            self.x = torch.nan_to_num(self.x, nan=0.0)
            print('Replaced masked nan values with 0.')
        else:
            print('No masked nan values detected.')

    def posterior_predictive_check(self,
            n_samples: int,
            x: Tensor,
            samples: Optional[Tensor] = None,
            simulator=None,
            num_workers: int = 16
        ) -> None:
        
        if not self.model:
            assert simulator, 'Pass a simulator to this function.'
        else:
            simulator = self.model.simulator
        
        if type(samples) is Tensor:
            samples_obj = Samples(samples)
        else:
            assert hasattr(self, 'posterior'), (
                'Since no posterior attribute exists for this instance to sample from, '
                'please pass a Tensor of samples to the function.'
            )
            samples = self.posterior.sample((n_samples,), x)
            assert type(samples) is Tensor 
            sample_obj = Samples(samples)

        simulator = process_simulator(
            simulator, sample_obj, is_numpy_simulator=False
        )
        theta, x = simulate_for_sbi(
            simulator=simulator,
            proposal=sample_obj,
            num_simulations=n_samples,
            num_workers=num_workers)
        
        for i in range(n_samples):
            print(f'Samples: {samples[i, :]}')
            smooth_map(x[i, :])
            plt.show()
    
    def run_simulation_based_calibration(self,
            num_posterior_samples: int = 1000,
            use_multidim_sbc: bool = False,
            num_sbc_samples: int = 200
        ) -> None:
        assert hasattr(self, 'posterior')

        indices = np.random.choice(self.theta.shape[0], num_sbc_samples, replace=False)
        theta = self.theta[indices]
        x = self.x[indices]

        ranks, dap_samples = run_sbc(
            theta,
            x,
            self.posterior,
            num_posterior_samples=1000,
            num_workers=1,
            reduce_fns=self.posterior.log_prob if use_multidim_sbc else 'marginals'
        )
        check_stats = check_sbc(
            ranks, theta, dap_samples,
            num_posterior_samples=num_posterior_samples
        )

        print(
            f"kolmogorov-smirnov p-values \ncheck_stats['ks_pvals'] = {check_stats['ks_pvals'].numpy()}"
        )
        print(
            f"c2st accuracies \ncheck_stats['c2st_ranks'] = {check_stats['c2st_ranks'].numpy()}"
        )
        print(
            f"c2st accuracies check_stats['c2st_dap'] = {check_stats['c2st_dap'].numpy()}"
        )

        f, ax = sbc_rank_plot(
            ranks=ranks,
            num_posterior_samples=num_posterior_samples,
            plot_type="hist",
            num_bins=None,  # heuristic for the number of bins
        )
        plt.show()

        # the tarp method returns the ECP values for a given set of alpha coverage levels.
        ecp, alpha = run_tarp(
            theta,
            x,
            self.posterior,
            references=None,  # will be calculated automatically.
            num_posterior_samples=num_posterior_samples,
        )

        # Similar to SBC, we can check then check whether the distribution of ecp is close to
        # that of alpha.
        atc, ks_pval = check_tarp(ecp, alpha)
        print(atc, "Should be close to 0")
        print(ks_pval, "Should be larger than 0.05")

        # Or, we can perform a visual check.
        from sbi.analysis.plot import plot_tarp

        plot_tarp(ecp, alpha)
        plt.show()