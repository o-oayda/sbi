import dynesty
import emcee
import numpy as np
import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import hpCNNEmbedding
from sbi.diagnostics import check_sbc, run_sbc, check_tarp, run_tarp
from sbi.analysis.plot import sbc_rank_plot
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.utils.user_input_checks import CustomPriorWrapper
import pickle
import healpy as hp
from numpy.typing import NDArray
from torch.types import Tensor
from typing import Optional
from dipolesbi.tools import Samples
from dipolesbi.tools.plotting import smooth_map
from dipolesbi.tools import Simulator 
import matplotlib.pyplot as plt
from sbi.analysis.plot import plot_tarp


class LikelihoodFreeInferer:
    def __init__(self, simulator: Optional[Simulator] = None) -> None:
        self.simulator = simulator
        self.posterior: NeuralPosterior | None = None

    def run_healpix_sbi(self,
            training_device: str = 'cuda',
            load_simulations_in_vram: bool = False,
            simulation_fraction: float = 1.0,
            nan_fill_value: float = 0.
    ) -> None:
        assert self.simulator is not None, 'Pass an instance of Simulator at init.'
        x = self.simulator.x; theta = self.simulator.theta
        prior = self.simulator.sbi_processed_prior

        assert (x is not None) and (theta is not None), (
            'Theta and x have not been assigned in the Simulator instance!'
        )
        assert isinstance(prior, CustomPriorWrapper), (
            "The simulator's prior has not been processed by sbi."
        )
        
        self.n_train_indices = int(simulation_fraction * len(x))
        print(
            f'Using {self.n_train_indices} simulations ({simulation_fraction*100}%)...'
        )
        
        # calling this again here should put the mean and variance attributes
        # on the right device, as well as the support boundaries (thanks to
        # the custom to method we defined)
        prior.to(training_device)

        # data process healpy maps
        x = self._check_for_mask_nans(x, nan_fill_value)
        self.nside = hp.npix2nside(x.shape[-1])
        
        embedding_net = hpCNNEmbedding(
            nside=self.nside,
            dropout_rate=0.2,
            out_channels_per_layer=[2, 4, 8, 16, 32, 64]
        )
        neural_posterior = posterior_nn(
            model="maf",
            embedding_net=embedding_net,
            z_score_x='structured'
        )
        inference = NPE(
            prior=prior,
            density_estimator=neural_posterior,
            device=training_device
        )

        # by specifying data_device = 'cpu', we can train on the GPU
        # while transferring data from host memory to VRAM
        self.inference = inference.append_simulations(
            theta[:self.n_train_indices],
            x[:self.n_train_indices],
            data_device='cpu' if not load_simulations_in_vram else 'cuda'
        )
        self.density_estimator = self.inference.train(show_train_summary=True)
        self.posterior = inference.build_posterior(
            self.density_estimator,
            prior=prior
        )
        print(self.posterior)

    def save_posterior(self, file_path: str) -> None:
        # be sure to transfer posterior to CPU, otherwise it royally fucks you
        # if you don't load it back on a machine with cuda
        self.posterior.to('cpu') # type: ignore
        print(f'Saving to {file_path}...')
        with open(file_path, "wb") as handle:
            pickle.dump(self.posterior, handle)

    def load_posterior(self, file_path: str) -> None:
        print(f'Opening {file_path}...')
        with open(file_path, "rb") as handle:
            self.posterior = pickle.load(handle)
        self.posterior.to('cpu')
        assert self.posterior is not None

    def sample_amortized_posterior(self,
            x_obs,
            n_samps: int = 10_000,
            **kwargs
        ) -> Tensor:
        assert self.posterior is not None, 'Posterior not infered or loaded.'
        return self.posterior.sample((n_samps,), x=x_obs, **kwargs)

    def _check_for_mask_nans(self, x: Tensor, fill_value: float = 0.) -> Tensor:
        assert x is not None, 'Simulator has no data!' 
        assert type(x) is Tensor, 'Convert simulator data to Tensor.'

        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=fill_value)
            print('Replaced masked nan values with 0.')
        else:
            print('No masked nan values detected.')

        return x

    def posterior_predictive_check(self,
            n_samples: int,
            x: Tensor,
            samples: Optional[Tensor] = None,
            num_workers: int = 16
        ) -> None:
        assert self.simulator is not None, 'Pass an instance of Simulator at init.'

        if type(samples) is Tensor:
            samples_obj = Samples(samples)
        else:
            assert self.posterior is not None, (
                'Since no posterior attribute exists for this instance to '
                'sample from, please pass a Tensor of samples to the function.'
            )
            samples = self.posterior.sample((n_samples,), x)
            assert type(samples) is Tensor 
            samples_obj = Samples(samples)

        _, x = self.simulator.make_batch_simulations(
            sbi_processed_prior=samples_obj, # type: ignore
            prior_returns_numpy=False,
            n_simulations=n_samples,
            n_workers=num_workers
        )
        
        for i in range(n_samples):
            print(f'Samples: {samples[i, :]}')
            smooth_map(x[i, :].numpy())
            plt.show()

    def run_simulation_based_calibration(self,
            num_posterior_samples: int = 1000,
            use_multidim_sbc: bool = False,
            num_sbc_samples: int = 200
        ) -> None:
        assert self.posterior is not None, 'Posterior not infered or loaded.'
        assert self.simulator is not None, 'Pass an instance of Simulator at init.'
        assert self.simulator._simulation_is_loaded()

        # simulator variables will not be None due to the above assertion
        indices = np.random.choice(
            self.simulator.theta.shape[0], # type: ignore 
            num_sbc_samples,
            replace=False
        )
        theta = self.simulator.theta[indices] # type: ignore
        x = self.simulator.x[indices] # type: ignore

        ranks, dap_samples = run_sbc(
            theta,
            x,
            self.posterior,
            num_posterior_samples=num_posterior_samples,
            num_workers=1,
            # should have log prob method as per NeuralPosterior docstring
            reduce_fns=self.posterior.log_prob if use_multidim_sbc else 'marginals' # type: ignore
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

        sbc_rank_plot(
            ranks=ranks,
            num_posterior_samples=num_posterior_samples,
            plot_type="hist",
            num_bins=None,  # heuristic for the number of bins
        )
        plt.show()

        # the tarp method returns the ECP values for a given set of alpha coverage levels.
        self.ecp, self.alpha = run_tarp(
            theta,
            x,
            self.posterior,
            references=None,  # will be calculated automatically.
            num_posterior_samples=num_posterior_samples,
        )

        # Similar to SBC, we can check then check whether the distribution of ecp is close to
        # that of alpha.
        self.ecp = self.ecp.to('cpu'); self.alpha = self.alpha.to('cpu')
        atc, ks_pval = check_tarp(self.ecp, self.alpha)
        print(atc, "Should be close to 0")
        print(ks_pval, "Should be larger than 0.05")

        plot_tarp(self.ecp, self.alpha)
        plt.show()

class LikelihoodBasedInferer:
    def __init__(self, prior, model):
        self.prior = prior
        self.model = model
        self.ndim = self.prior.ndim

    def run_mcmc(self,
        nwalkers: int = 32,
        n_steps: int = 2000,
        burn_in: int = 100
    ) -> None:
        '''
        Run MCMC with emcee and save posterior in samples attribute.
        '''

        def log_prob(Theta):
            log_prior = self.prior.log_prior_likelihood(Theta)
            if not np.isfinite(log_prior):
                return -np.inf
            return log_prior + self.model.log_likelihood(Theta)

        pos = np.zeros((nwalkers, self.ndim))
        for i in range(0, nwalkers):
            unifs = np.random.rand(self.ndim)
            pos[i, :] = self.prior.prior_transform(unifs)

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
            self.model.log_likelihood,
            self.model.prior_transform,
            **{
                'ndim': self.ndim,
               'sample': sample_method,
               **kwargs
            }
        )

        dsampler.run_nested(print_progress=print_info)
        self.model_evidence = dsampler.results.logz[-1] # type: ignore
        print('Model evidence: {:.2f}'.format(self.model_evidence))
        self.dresults = dsampler.results
      
