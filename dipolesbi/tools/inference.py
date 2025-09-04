from anesthetic import NestedSamples
import blackjax
from blackjax.types import PRNGKey
import dynesty
import emcee
from joblib.parallel import Parallel, delayed
import numpy as np
import torch
from sbi.inference import NPE, NLE, NRE
from sbi.neural_nets import likelihood_nn, posterior_nn
from sbi.neural_nets import classifier_nn
from sbi.neural_nets.embedding_nets import hpCNNEmbedding
from sbi.diagnostics import check_sbc, run_sbc, check_tarp, run_tarp
from sbi.analysis.plot import sbc_rank_plot
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.utils.user_input_checks import CustomPriorWrapper
import pickle
import healpy as hp
from numpy.typing import NDArray
from torch.types import Tensor
from typing import Callable, Optional, Literal
from dipolesbi.tools import Samples
from dipolesbi.tools.models import LikelihoodBasedModel
from dipolesbi.tools.plotting import smooth_map
from dipolesbi.tools import Simulator 
import matplotlib.pyplot as plt
from sbi.analysis.plot import plot_tarp
from tqdm import tqdm
from dipolesbi.tools.priors import Prior
import ultranest
import jax
from jax import numpy as jnp
from dipolesbi.tools.priors_jax import JaxPrior
from dipolesbi.tools.priors_np import DipolePriorNP


class LikelihoodFreeInferer:
    def __init__(self, simulator: Optional[Simulator] = None) -> None:
        self.simulator = simulator
        self.posterior: NeuralPosterior | None = None

    def run_healpix_sbi(self,
            estimator_type: Literal['NPE', 'NLE', 'NRE'] = 'NPE',
            flow_type: Literal['maf', 'nsf'] = 'maf',
            training_device: str = 'cuda',
            load_simulations_in_vram: bool = False,
            simulation_fraction: float = 1.0,
            n_rounds: int = 1,
            x0_multi: Optional[Tensor] = None,
            multiround_workers: int = 32,
            nan_handle_method: Literal['fill', 'truncate'] = 'fill',
            nan_fill_value: float = 0.,
            z_score_x: Literal['structured', 'independent'] | None = None
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

        assert n_rounds >= 1, 'Must specify at least 1 round of inference.'
        if n_rounds > 1:
            assert self.simulator.simulation_model is not None, (
                'In multi-round inference, the simulator must have a data-generating '
                'model to allow inference on subsequent rounds.'
            )
            assert x0_multi is not None, (
                'In multi-round inference, a ground observation x0_multi must be set.'
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
        self.nside = hp.npix2nside(x.shape[-1])
        x = self._check_for_mask_nans(
            x,
            nan_handle_method=nan_handle_method, 
            fill_value=nan_fill_value
        )

        if estimator_type == 'NPE':
            embedding_net = hpCNNEmbedding(
                nside=self.nside,
                dropout_rate=0.2,
                out_channels_per_layer=[2, 4, 8, 16, 32, 64]
            )
            neural_posterior = posterior_nn(
                model=flow_type,
                embedding_net=embedding_net,
                z_score_x=z_score_x
            )
            inference = NPE(
                prior=prior,
                density_estimator=neural_posterior,
                device=training_device
            )
        elif estimator_type == 'NLE':
            # assert self.nside <= 4, (
            #     'Embedding network not supported for NLE; '
            #     'be sure to keep nside low'
            # )
            neural_posterior = likelihood_nn(
                model=flow_type,
                z_score_x=z_score_x,
                z_score_theta='independent'
            )
            inference = NLE(
                prior=prior,
                density_estimator=neural_posterior,
                device=training_device
            )
            # raise NotImplementedError 
        elif estimator_type == 'NRE':
            embedding_net = hpCNNEmbedding(
                nside=self.nside,
                dropout_rate=0.2,
                out_channels_per_layer=[2, 4, 8, 16, 32, 64]
            )
            neural_classifier = classifier_nn(
                model="resnet",
                embedding_net_x=embedding_net,
                z_score_x=z_score_x,
                z_score_theta='independent'
            )
            inference = NRE(
                prior=prior,
                classifier=neural_classifier,
                device=training_device
            )
        else:
            raise Exception(f'Choose from NPE, NLE, or NRE ({estimator_type} chosen).')

        # by specifying data_device = 'cpu', we can train on the GPU
        # while transferring data from host memory to VRAM
        posteriors = []
        for i in range(n_rounds):
            print(f'Starting round {i+1} of inference...')
            n_sims = x.shape[0]
            if i > 0:
                # again because I keep getting shat on by this fucking library
                prior.to('cpu')
                proposal.to('cpu')

                theta, x = self.simulator.make_batch_simulations(
                    n_simulations=n_sims, # set from input sim count
                    n_workers=multiround_workers,
                    proposal=proposal, # type: ignore
                    simulation_batch_size=100
                )
            self.inference = inference.append_simulations(
                x=x[:self.n_train_indices],
                theta=theta[:self.n_train_indices],
                data_device='cpu' if not load_simulations_in_vram else 'cuda'
            )
            self.density_estimator = self.inference.train(
                show_train_summary=True,
                stop_after_epochs=30
            )

            if estimator_type == 'NPE':
                self.posterior = inference.build_posterior(
                    self.density_estimator, # type: ignore
                    prior=prior
                )
            elif estimator_type in ['NLE', 'NRE']:
                self.posterior = inference.build_posterior(
                    density_estimator=self.density_estimator, # type: ignore
                    prior=prior,
                    sample_with='mcmc',
                    mcmc_method='slice_np_vectorized'
                )
            else:
                raise Exception(f'Unknown estimator type: {estimator_type}')

            if n_rounds > 1:
                posteriors.append(self.posterior)
                proposal = self.posterior.set_default_x(x0_multi) # type: ignore

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
        self.posterior.to('cpu') # type: ignore
        assert self.posterior is not None

    def sample_amortized_posterior(self,
            x_obs,
            n_samps: int = 10_000,
            **kwargs
        ) -> Tensor:
        assert self.posterior is not None, 'Posterior not infered or loaded.'
        return self.posterior.sample((n_samps,), x=x_obs, **kwargs)

    def plot_loss_curve(self) -> None:
        summary = self.inference.summary
        training_loss = summary['training_loss']
        validation_loss = summary['validation_loss']
        
        plt.plot(training_loss, c='tab:blue', label='Training loss')
        plt.plot(validation_loss, c='tab:orange', label='Validation loss')
        plt.legend()
        plt.show()

    def _check_for_mask_nans(
            self, 
            x: Tensor, 
            nan_handle_method: Literal['fill', 'truncate'] = 'fill', 
            fill_value: float = 0.
    ) -> Tensor:
        assert x is not None, 'Simulator has no data!' 
        assert type(x) is Tensor, 'Convert simulator data to Tensor.'

        if torch.isnan(x).any():

            if nan_handle_method == 'fill':
                x = torch.nan_to_num(x, nan=fill_value)
                print('Replaced masked nan values with 0.')
            elif nan_handle_method == 'truncate':
                valid_columns = ~torch.isnan(x).any(dim=0)
                x = x[:, valid_columns]
                print('Truncated dataset by removing masked nan values.')

        else:
            print('No masked nan values detected.')

        return x

    def _quick_simulate(self,
            theta: Tensor,
            model_callable: Callable[..., NDArray],
            custom_prior: Prior
    ) -> Tensor:
        n_simulations = theta.shape[0]
        simulation_batch_size = 1
        n_batches = n_simulations // simulation_batch_size # batch size of 1 by default in simulate_for_sbi
        theta_np = theta.detach().cpu().numpy()
        batches = np.array_split(theta_np, n_batches, axis=0)
        n_workers = n_simulations

        def model_wrapper(Theta: NDArray) -> NDArray:
            mapping = {
                kwarg: np.float64(Theta[i]) for i, kwarg in enumerate(
                    custom_prior.simulator_kwargs
                )
            }
            return model_callable(**mapping)

        def simulation_wrapper(Theta: NDArray) -> NDArray:
            xs = list(map(model_wrapper, Theta))
            return np.stack(xs)
       
        simulation_outputs: list[NDArray] = [ # type: ignore
            xx
            for xx in tqdm(
                Parallel(return_as='generator', n_jobs=n_workers)(
                    delayed(simulation_wrapper)(batch)
                    for batch in batches
                ),
                total=n_simulations
            )
        ]

        x = np.vstack(simulation_outputs)
        return torch.as_tensor(x)

    def posterior_predictive_check(self,
            n_samples: int,
            x_real: NDArray,
            mask: NDArray,
            model_callable: Callable[..., NDArray],
            samples: Optional[Tensor] = None
        ) -> None:
        assert n_samples <= 10, (
            'n_workers = n_samples by design; avoid setting too many samples'
        )
        assert self.posterior is not None, 'Load a posterior distribution first.'
        samples = self.posterior.sample((n_samples,), torch.as_tensor(x_real))
        custom_prior = self.posterior.prior.custom_prior # type: ignore
        assert isinstance(custom_prior, Prior)
        assert type(samples) is Tensor 
        samples_obj = Samples(samples)

        x_sim = self._quick_simulate(
            theta=samples_obj.sample((n_samples,)),
            model_callable=model_callable,
            custom_prior=custom_prior
        )
        
        x_real[mask] = np.nan
        real_smooth_map = smooth_map(x_real, only_return_data=True)
        sub = f'{n_samples + 1}1'
        initial_subplot = int( sub + str(1) )
        hp.projview(
            real_smooth_map,
            sub=initial_subplot, 
            title='Real data', 
            nest=True,
            cb_orientation='vertical'
        )

        for i in range(1, n_samples+1):
            smooth_map_to_plot = smooth_map(
                x_sim[i-1, :].numpy(), 
                only_return_data=True
            )

            sample_readout = f'Sample {i}:'
            for idx, param in enumerate(samples[i-1, :]):
                kwarg = custom_prior.simulator_kwargs[idx]
                sample_readout += f'\n\t{kwarg}: {param:.4g}'
            print(sample_readout)

            hp.projview(
                smooth_map_to_plot, 
                sub=int( sub + str(i+1) ), 
                title=f'Predictive {i}',
                nest=True,
                cb_orientation='vertical'
            )

        # plt.show()

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

class JaxNestedSampler:
    def __init__(self, lnlike: Callable, prior: JaxPrior) -> None:
        self.lnlike = lnlike
        self.prior = prior
        self.ndim = self.prior.ndim

    def setup(
            self,
            rng_key: PRNGKey, 
            n_live: int = 500,
            n_delete: int = 50
    ) -> None:
        n_inner_steps = self.ndim * 5
        self.n_delete = n_delete
        self.rng_key, self.prior_key = jax.random.split(rng_key)
        self.particles = self.prior.get_initial_live_samples(self.prior_key, n_live)

        self.nested_sampler = blackjax.nss(
            logprior_fn=self.prior.log_prob,
            loglikelihood_fn=self.lnlike,
            num_delete=n_delete,
            num_inner_steps=n_inner_steps,
        )
        print(f"Initialised nested sampler with {n_live} live points.")

        self._jit_functions()

    def _jit_functions(self) -> None:
        self.init_fn = jax.jit(self.nested_sampler.init)
        self.step_fn = jax.jit(self.nested_sampler.step)
        print("Functions compiled - ready to run!")

    def run(self) -> NestedSamples:
        print("Running nested sampling...")
        live = self.init_fn(self.particles)
        dead = []

        with tqdm(desc="Dead points", unit=" dead points") as pbar:
            while not live.logZ_live - live.logZ < -3:  # Convergence criterion
                self.rng_key, subkey = jax.random.split(self.rng_key, 2)
                live, dead_info = self.step_fn(subkey, live)
                dead.append(dead_info)
                pbar.update(self.n_delete)

        dead = blackjax.ns.utils.finalise(live, dead)
        columns = self.prior.simulator_kwargs
        data = jnp.vstack([dead.particles[key] for key in columns]).T # type: ignore

        self.nested_samples = NestedSamples(
            data,
            logL=dead.loglikelihood,
            logL_birth=dead.loglikelihood_birth,
            columns=columns,
            logzero=jnp.nan,
        )

        print(
            f"Log Evidence: {self.nested_samples.logZ():.2f} "
            f"± {self.nested_samples.logZ(100).std():.2f}" # type: ignore
        )
        
        return self.nested_samples

class NotShitLikelihoodBasedInferer:
    def __init__(
            self,
            lnlike: Callable[[dict[str, NDArray]], NDArray],
            prior: DipolePriorNP,
            data: NDArray
    ) -> None:
        self.lnlike = lnlike
        self.prior = prior
        self.data = data

    def run_ultranest(self) -> None:
        def log_likelihood_wrapper(theta: NDArray) -> NDArray:
            theta_dict: dict[str, NDArray] = {}
            for i in range(self.prior.ndim):
                theta_dict[self.prior.simulator_kwargs[i]] = theta[:, i]
            return self.lnlike(theta=theta_dict) # type: ignore fuck off you fucking piece of shit

        def prior_transform_wrapper(unifcube_samples: NDArray) -> NDArray:
            unif_dict: dict[str, NDArray] = {}
            for i in range(self.prior.ndim):
                unif_dict[self.prior.prior_names[i]] = unifcube_samples[:, i]
            tformed_samples = self.prior.transform(unif_dict)
            tformed_list = [val for val in tformed_samples.values()]
            return np.stack(tformed_list, axis=1)

        self.ultranest_sampler = ultranest.ReactiveNestedSampler(
            param_names=self.prior.prior_names,
            loglike=log_likelihood_wrapper,
            transform=prior_transform_wrapper,
            **{
                'log_dir': 'ultranest_logs',
                'resume': 'subfolder',
                'vectorized': True
            }
        )

        self.results = self.ultranest_sampler.run()
        self.ultranest_sampler.print_results()

        try:
            self.ultranest_sampler.plot()
        except ValueError as e:
            print(e)
        
        if self.results is not None:
            self._samples = self.results['samples']
            self.log_bayesian_evidence = self.results['logz']
            self.log_bayesian_evidence_err = self.results['logzerr']
        else:
            raise Exception('Ultranest results are undefined.')

class LikelihoodBasedInferer:
    def __init__(self, data: NDArray, model: LikelihoodBasedModel):
        self.model = model
        self.prior = model.prior
        self.data = data
        self.ndim = self.model.ndim

    def run_mcmc(self,
        nwalkers: int = 32,
        n_steps: int = 2000,
        burn_in: int = 100
    ) -> None:
        '''
        Run MCMC with emcee and save posterior in samples attribute.
        '''

        def log_prob(Theta):
            log_prior = self.model.log_prior_likelihood(Theta)
            if not np.isfinite(log_prior):
                return -np.inf
            return log_prior + self.model.log_likelihood(self.data, Theta)

        pos = np.zeros((nwalkers, self.ndim))
        for i in range(0, nwalkers):
            unifs = np.random.rand(self.ndim)
            pos[i, :] = self.model.prior_transform(unifs)

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
        def log_likelihood_wrapper(theta: NDArray) -> NDArray:
            return self.model.log_likelihood(data=self.data, theta=theta)

        dsampler = dynesty.NestedSampler(
            log_likelihood_wrapper,
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
      

    def run_ultranest(self,
            step: bool = False,
            n_steps: int | None = None,
            reactive_sampler_kwargs: dict = {},
            run_kwargs: dict = {}
        ) -> None:
        '''
        Perform Nested Sampling using ultranest's `ReactiveNestedSampler`.
        Results are saved in the results attribute.

        :param step: Specify whether or not to use the random step method.
        :param n_steps: If the random step method is specified, this is the
            number of steps as used by `SliceSampler`.
        :param reactive_sampler_kwargs: Keyword arguments for the
            `ReactiveNestedSampler`.
        :param run_kwargs: Keyword arguments for the sampler's run method.
        '''
        def log_likelihood_wrapper(theta: NDArray) -> NDArray:
            return self.model.log_likelihood(data=self.data, theta=theta)

        self.ultranest_sampler = ultranest.ReactiveNestedSampler(
            param_names=self.prior.prior_names,
            loglike=log_likelihood_wrapper,
            transform=self.model.prior_transform,
            **{
                'log_dir': 'ultranest_logs',
                'resume': 'subfolder',
                'vectorized': True,
                **reactive_sampler_kwargs
            }
        )

        # if step:
        #     self._switch_to_step_sampling(n_steps)
        
        self.results = self.ultranest_sampler.run(**run_kwargs)
        self.ultranest_sampler.print_results()

        # there is an issue with ultranest plotting when the log likelihood is
        # very negative (e.g. for the point-by-point likelihood)
        # this catches the ValueError raised
        try:
            self.ultranest_sampler.plot()
        except ValueError as e:
            print(e)
        
        if self.results is not None:
            self._samples = self.results['samples']
            self.log_bayesian_evidence = self.results['logz']
            self.log_bayesian_evidence_err = self.results['logzerr']
        else:
            raise Exception('Ultranest results are undefined.')
