from maps import SkyMap
from priors import DipolePrior
import dynesty
import emcee
import numpy as np
import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import hpCNNEmbedding
from sbi.utils.user_input_checks import process_prior

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
        '''
        Begin the nested sampling process and return the results upon
        completion.
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

    def run_sbi(self,
            n_simulations: int = 2000,
            device: str = 'cpu',
            **mask_kwargs
    ) -> None:
        prior = DipolePrior(
            mean_count_range=self.mean_count_range,
            amplitude_range=self.amplitude_range,
            longitude_range=self.longitude_range,
            latitude_range=self.latitude_range
        )
        self.theta, self.x = SkyMap().batch_simulator(
            prior, n_samples=n_simulations, **mask_kwargs
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