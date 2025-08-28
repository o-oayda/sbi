from dataclasses import asdict
from operator import pos
from typing import Any, Callable, Optional
from anesthetic import NestedSamples
from blackjax.types import PRNGKey
import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt
import healpy as hp
from scipy.stats import norm
from dipolesbi.tools.configs import NLEConfig, TrainingConfig
from dipolesbi.tools.dataloader import split_train_val, split_train_val_dict
from dipolesbi.tools.healpix_helpers import build_funnel_steps
from dipolesbi.tools.inference import JaxNestedSampler, LikelihoodBasedInferer
from dipolesbi.tools.maps import SimpleDipoleMap
from surjectors.util import named_dataset
from dipolesbi.tools.models import DipolePoisson
from dipolesbi.tools.neural_flows import MAFNeuralLikelihood, MAFSurjectiveNeuralLikelihood
from dipolesbi.tools.priors import DipolePrior
from dipolesbi.tools.priors_jax import DipolePriorJax
from dipolesbi.tools.simulator import Simulator
from dipolesbi.tools.transforms import HaarWaveletTransform, HealpixSOPyramid, learn_transformation
from dipolesbi.tools.utils import jax_cart2sph, jax_sph2cart
import torch
from jax import lax


class MultiRoundInferer:
    def __init__(
            self, 
            rng_key: PRNGKey,
            initial_proposal: DipolePriorJax,
            simulator_function: Callable[[PRNGKey, dict[str, jnp.ndarray], bool], jnp.ndarray],
            reference_observation: jnp.ndarray,
            reference_theta: Optional[dict[str, jnp.ndarray]] = None,
            simulation_budget: int = 50_000, 
            n_rounds: int = 10,
            nle_config: NLEConfig = NLEConfig.standard(),
            train_config: TrainingConfig = TrainingConfig()
    ) -> None:
        self.rng_key = rng_key
        self.simulation_budget = simulation_budget
        self.n_rounds = n_rounds
        self.nle_config = nle_config
        self.train_config = train_config
        self.simulations_per_round = self.simulation_budget // self.n_rounds
        self.initial_proposal = initial_proposal
        self.initial_proposal.change_kwarg('N', 'mean_density')
        self.simulator_function = simulator_function
        self.data_ndim = reference_observation.shape[-1]
        self.nside = hp.npix2nside(self.data_ndim)
        self.reference_observation = reference_observation[None, :]
        self.reference_theta = reference_theta
        self.sample_posterior_seed = 0
        np.random.seed(42) # hopefully to keep ultranest results deterministic
        
        self.nle_config = nle_config
        self.train_config = train_config

        self.data_mean = None
        self.data_std = None
        self.theta_mean = None
        self.theta_std = None

        self.all_data = jnp.full((self.simulation_budget, self.data_ndim), jnp.nan)
        self.all_theta = {
            'mean_density': jnp.full((self.simulation_budget,), jnp.nan),
            'observer_speed': jnp.full((self.simulation_budget,), jnp.nan),
            'dipole_longitude': jnp.full((self.simulation_budget,), jnp.nan),
            'dipole_latitude': jnp.full((self.simulation_budget), jnp.nan)
        }

    def run(self):
        current_key = self.rng_key

        for round_idx in range(self.n_rounds):
            self.current_round = round_idx

            current_key, proposal_key, sim_key, train_key = jax.random.split(
                current_key, 4
            )
            theta = self._sample_proposal(
                key=proposal_key,
                n_samples=self.simulations_per_round,
                use_initial=True if round_idx == 0 else False
            )

            print('Generating simulations...')
            data = self._generate_simulations(sim_key, theta)
            self._add_to_simulation_pool(data, theta)
            self.trn_set, self.val_set = self._make_train_val_set()
            del data; del theta

            self.nle = self._instantiate_nle()


            print('Starting training...')
            self.nle.train(
                hk.PRNGSequence(train_key),
                self.trn_set, 
                self.val_set,
                **asdict(self.train_config)
            )
            self.nle.plot_loss_curve()
            self._inspect_learned_likelihood(train_key)

            self._compute_posterior()

        self._benchmark_classic()

    def _inspect_learned_likelihood(self, rng_key: PRNGKey, n_repeats: int = 50_000) -> None:
        sample_key, simulate_key = jax.random.split(rng_key)

        assert self.reference_theta is not None

        samples = self.nle.sample_likelihood_func(
            sample_key,
            theta0=jnp.repeat(
                self._normalise_theta(self.reference_theta, in_ns=True)[None, :],
                repeats=n_repeats,
                axis=0
            )
        )
        true_mean_likelihood = self.simulator_function(
            simulate_key,
            self.reference_theta,
            False
        )
        self.mean_samples = self._unnormalise_data(samples).mean(axis=0)

        hp.projview(
            self.mean_samples, nest=True, graticule=True, sub=211,
            title='NLE mean P(D | theta)'
        )
        hp.projview(
            true_mean_likelihood, nest=True, graticule=True, sub=212, 
            title='True mean P(D | theta)'
        )
        plt.show()

    def _benchmark_classic(self) -> None:
        self.classic_prior = DipolePrior(
            mean_count_range=[
                float(self.initial_proposal.prior_dict['N']['low_range']),
                float(self.initial_proposal.prior_dict['N']['high_range']),
            ]
        )
        self.classic_prior.change_kwarg('N', 'mean_density')
        mask_map = np.ones(self.data_ndim, dtype=np.bool_)
        classic_model = DipolePoisson(
            self.classic_prior,
            nside=self.nside,
            mask_map=mask_map
        )
        self.classic_inferer = LikelihoodBasedInferer(
            np.asarray(self.reference_observation.squeeze()),
            classic_model
        )
        self.classic_inferer.run_ultranest()

        print(
            f"NLE Log Evidence: {self.nested_samples.logZ():.2f} "
            f"± {self.nested_samples.logZ(100).std():.2f}" # type: ignore
        )
        print(
            f"Classic Log Evidence: {self.classic_inferer.log_bayesian_evidence:.2f} "
            f"± {self.classic_inferer.log_bayesian_evidence_err:.2f}" # type: ignore
        )
        diff = self.nested_samples.logZ() - self.classic_inferer.log_bayesian_evidence # type: ignore
        sigma = (
            np.abs(diff) # type: ignore
          / np.sqrt(
                self.nested_samples.logZ(100).std()**2  # type: ignore
              + self.classic_inferer.log_bayesian_evidence_err**2 # type: ignore
            ) 
        )
        print(f'Tension: {sigma}')

    def _add_to_simulation_pool(
            self,
            data: jnp.ndarray,
            theta: dict[str, jnp.ndarray]
    ) -> None:
        start = self.current_round * self.simulations_per_round
        end = (self.current_round + 1) * self.simulations_per_round

        print(f'Adding data from {start} to {end}...')
        self.all_data = self.all_data.at[start:end, :].set(data)
        self.all_theta = self._write_contiguous(self.all_theta, theta, start)

    def _write_contiguous(self, storage, batch, start: int):
        def put(dst, src):
            # write src into dst at [start:start+src.shape[0]] along axis 0
            idx = (start,) + (0,) * (dst.ndim - 1)
            return lax.dynamic_update_slice(dst, src, idx)
        return jax.tree.map(put, storage, batch)

    def _compute_posterior(self) -> None:
        self.rng_key, ns_key = jax.random.split(self.rng_key)

        x0, log_det_jac = self._normalise_data(self.reference_observation)

        def lnlike_jax(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
            theta = self._normalise_theta(params, in_ns=True)
            log_like = self.nle.evaluate_lnlike(theta[None, :], x0)
            log_like += log_det_jac
            return log_like.squeeze()

        self.jax_ns = JaxNestedSampler(lnlike_jax, self.initial_proposal)
        self.jax_ns.setup(ns_key, n_live=1000, n_delete=50)
        self.nested_samples = self.jax_ns.run()

        kinds = {'lower': 'kde_2d', 'diagonal': 'hist_1d', 'upper': 'scatter_2d'}
        self.nested_samples.plot_2d(
            self.initial_proposal.simulator_kwargs,
            kinds=kinds
        )
        plt.show()

    def _make_train_val_set(self) -> tuple[named_dataset, named_dataset]:
        (trn_data, val_data), (trn_theta, val_theta) = self._split_train_val(
            self.all_data, self.all_theta
        )

        self._compute_norm_stats(trn_data, trn_theta)

        norm_train_data, _ = self._normalise_data(trn_data)
        norm_val_data, _ = self._normalise_data(val_data)
        norm_train_theta = self._normalise_theta(trn_theta)
        norm_val_theta = self._normalise_theta(val_theta)

        assert not jnp.any(jnp.isnan(norm_train_data))
        assert not jnp.any(jnp.isnan(norm_train_theta))
        assert not jnp.any(jnp.isnan(norm_val_data))
        assert not jnp.any(jnp.isnan(norm_train_data))

        trn_set = named_dataset(norm_train_data, norm_train_theta)
        val_set = named_dataset(norm_val_data, norm_val_theta)

        return trn_set, val_set

    def _split_train_val(
            self, 
            data: jnp.ndarray, 
            theta: dict[str, jnp.ndarray]
    ) -> tuple[
        tuple[jnp.ndarray, jnp.ndarray], 
        tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]
    ]:
        cur_max_idx = (self.current_round+1) * self.simulations_per_round
        cur_data = data[:cur_max_idx, :]
        cur_theta = jax.tree_map(lambda x: x[:cur_max_idx], theta)
        return split_train_val_dict(cur_data, cur_theta)

    def _compute_norm_stats(
            self,
            train_data: jnp.ndarray, 
            train_theta: dict[str, jnp.ndarray]
    ) -> None:
        # per pixel mean and std across all batches
        self.data_mean = jnp.nanmean(train_data, axis=0)
        self.data_std = jnp.nanstd(train_data, axis=0) + 1e-8

        mean_nbar = jnp.nanmean(train_theta['mean_density'])
        std_nbar = jnp.nanstd(train_theta['mean_density'])
        mean_v = jnp.nanmean(train_theta['observer_speed'])
        std_v = jnp.nanstd(train_theta['observer_speed'])
        self.theta_mean = jnp.asarray([mean_nbar, mean_v, 0, 0])
        self.theta_std = jnp.asarray([std_nbar, std_v, 1, 1])

    def _normalise_theta(
        self,
        theta: dict[str, jnp.ndarray],
        in_ns: bool = False
    ) -> jnp.ndarray:
        lon = jnp.deg2rad(theta['dipole_longitude'])
        colat = jnp.pi / 2 - jnp.deg2rad(theta['dipole_latitude'])
        x, y, z = jax_sph2cart(lon, colat)

        assert self.theta_mean is not None
        assert self.theta_std is not None

        t_transformed = jnp.stack(
            [
                (theta['mean_density'] - self.theta_mean[0]) / self.theta_std[0],
                (theta['observer_speed'] - self.theta_mean[1]) / self.theta_std[1],
                x, y, z
            ],
            axis=1 if not in_ns else 0
        )
        return t_transformed

    def _normalise_data(self, data: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Normalise only using stats computed from the training data.
        '''
        assert self.data_mean is not None
        assert self.data_std is not None

        data_norm = (data - self.data_mean) / self.data_std
        log_det_jac = - jnp.log(self.data_std).sum()

        return data_norm, log_det_jac

    def _unnormalise_data(self, z: jnp.ndarray) -> jnp.ndarray:
        data = z * self.data_std + self.data_mean
        return data

    def _sample_proposal(
            self,
            key: PRNGKey,
            n_samples: int, 
            use_initial: bool = True,
            learned_fraction: float = 0.7,
            initial_fraction: float = 0.3
    ) -> dict[str, jnp.ndarray]:
        if use_initial:
            init_keys = jax.random.split(key, n_samples)
            print('Sampling from initial proposal...')
            prior_samples = jax.vmap(self.initial_proposal.sample)(init_keys)
        else:
            n_prior = int(0.3 * n_samples)
            n_posterior = self.simulations_per_round - n_prior
            print(
                f'Sampling from learned posterior {learned_fraction} '
                f'and initial proposal {initial_fraction}...'
            )
            posterior_samples = self.nested_samples.sample(
                n=n_posterior,
                random_state=self.sample_posterior_seed
            )
            self.sample_posterior_seed += 1
            posterior_samples = self._reformat_samples(posterior_samples)

            init_keys = jax.random.split(key, n_prior)
            initial_samples = jax.vmap(self.initial_proposal.sample)(init_keys)

            prior_samples = jax.tree_map(
                lambda a, b: jnp.concatenate([a, b], axis=0),
                posterior_samples,
                initial_samples
            )
            
        return prior_samples

    def _reformat_samples(
        self,
        samples: NestedSamples
    ) -> dict[str, jnp.ndarray]:
        samples_dict = samples.to_dict(orient='list')

        del samples_dict['logL']
        del samples_dict['logL_birth']
        del samples_dict['nlive']

        return {key: jnp.asarray(val) for key, val in samples_dict.items()}

    def _generate_simulations(
            self, 
            key: PRNGKey,
            theta: dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        sample_keys = jax.random.split(key, self.simulations_per_round)
        simulations = jax.vmap(self.simulator_function)(sample_keys, theta)
        return simulations

    def _instantiate_nle(self) -> MAFSurjectiveNeuralLikelihood:
        config_dict = asdict(self.nle_config)
        config_dict.pop('flow_type', None)

        nle = MAFSurjectiveNeuralLikelihood(
            self.data_ndim,
            **config_dict
        )
        return nle


if __name__ == '__main__':
    rng_key = jax.random.PRNGKey(42)
    n = 20_000
    torch_training_device = 'cuda'

# healpix Poisson data
    NSIDE = 32
    last_nside = 2
    ndim=12 * NSIDE**2
    TOTAL_SOURCES = 1_920_000
    MEAN_DENSITY = TOTAL_SOURCES / hp.nside2npix(NSIDE)
    OBSERVER_SPEED = 2.
    DIPOLE_LONGITUDE = 215.
    DIPOLE_LATITUDE = 40.
    N_EVIDENCE = 30
    mask_map = np.ones(ndim, dtype=np.bool_)
    theta0 = jnp.repeat(
        jnp.asarray(
            [[MEAN_DENSITY, OBSERVER_SPEED, DIPOLE_LONGITUDE, DIPOLE_LATITUDE]]
        ),
        repeats=n,
        axis=0
    )

    dipole = SimpleDipoleMap(NSIDE)    
    mean_count_range = [0.99*MEAN_DENSITY, 1.01*MEAN_DENSITY]
    prior = DipolePrior(mean_count_range=mean_count_range)
    jax_prior = DipolePriorJax(mean_count_range=mean_count_range)
    prior.change_kwarg('N', 'mean_density')
    simulator = Simulator(prior, dipole.generate_dipole)
    prior_samples, y = simulator.make_batch_simulations(
        n_simulations=n,
        n_workers=32,
        simulation_batch_size=1000
    )
    y = y + (np.random.rand(*y.shape) - 0.5)
    prior_samples = jnp.asarray(prior_samples)
    y = jnp.asarray(y)
    (y_tr, y_val), (t_tr, t_val) = split_train_val(y, prior_samples)

    y_mean = y_tr.mean(axis=0); y_std = y_tr.std(axis=0) + 1e-8
# y_mean = y_tr.mean(); y_std = y_tr.std(axis=1).mean()

# shitty at nside=1
    last_npix = 12 * last_nside**2
    tform = HaarWaveletTransform()
    def normalise_y(y):
        z, logdet = tform.cycle_healpix_tree(y, last_nside=last_nside)
        return z
    unnormalise_y = lambda input: input

    detail_sizes = []
    ns = NSIDE
    while ns > last_nside:
        ns //= 2
        npi = 12 * ns**2
        detail_sizes.append(3*npi)
    block_lengths = [last_npix] + detail_sizes # a coeffs + 3 details per level
    bounds = np.cumsum(block_lengths)
    starts  = np.concatenate(([0], bounds[:-1]))
    blocks = [(int(s), int(e)) for (s, e) in zip(starts[1:], bounds[1:])]
    assert 12 * NSIDE**2 == sum(block_lengths)
    steps = build_funnel_steps(n_coarse=12*last_nside**2, detail_lengths=block_lengths[1:])

# log_det_jac = lambda
# normalise_y = lambda input: (input - y_mean) / y_std
# unnormalise_y = lambda input: y_std * input + y_mean
# log_det_jac = lambda y: - jnp.log(y_std) * jnp.ones(y.shape[-1])

    mean_nbar = t_tr[:, 0].mean()
    std_nbar = t_tr[:, 0].std()
    mean_v = t_tr[:, 1].mean()
    std_v = t_tr[:, 1].std()
    t_mean = jnp.asarray([mean_nbar, mean_v, 0, 0])
    t_std = jnp.asarray([std_nbar, std_v, 1, 1])

    def transform_t(t):
        lon = jnp.deg2rad(t[..., 2])
        colat = jnp.pi / 2 - jnp.deg2rad(t[..., 3])
        x, y, z = jax_sph2cart(lon, colat)
        
        t_norm = (t - t_mean) / t_std
        t_transformed = jnp.stack(
            [
                t_norm[:, 0],
                t_norm[:, 1],
                x, y, z
            ],
            axis=1
        )
        return t_transformed

    def transform_theta_jax(
            params: dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        lon = jnp.deg2rad(params['phi'])
        colat = jnp.pi / 2 - jnp.deg2rad(params['theta'])
        x, y, z = jax_sph2cart(lon, colat)

        N_norm = (params['N'] - t_mean[0]) / t_std[0]
        D_norm = (params['D'] - t_mean[1]) / t_std[1]

        t_transformed = jnp.stack(
            [
                N_norm,
                D_norm,
                x, y, z
            ],
        )
        return t_transformed

    def untransform_t(t):
        x, y, z = t[:, 2], t[:, 3], t[:, 4]
        phi, theta = jax_cart2sph(x, y, z)
        lon = jnp.rad2deg(phi)
        lat = 90. - jnp.rad2deg(theta)

        t = t * t_std + t_mean
        t_untransformed = jnp.stack(
            [
                t[:, 0],
                t[:, 1],
                lon,
                lat
            ],
            axis=1
        )
        return t_untransformed

    def inspect_coeffs(z: jnp.ndarray, batchwise: Optional[list[int]] = None) -> None:
        ns = NSIDE
        n_coarse = hp.nside2npix(last_nside)
        start = n_coarse
        end = n_coarse

        if not batchwise:
            n_iters = 1
        else:
            n_iters = len(batchwise)

        for i in range(n_iters):
            while ns > last_nside:
                ns //= 2
                n_details = 3 * 12 * ns**2 
                end += n_details

                if not batchwise:
                    details = z[:, start:end]
                    bins = list(np.linspace(-5, 5, 100))
                else:
                    details = z[batchwise[i], start:end]
                    bins = list(np.linspace(-5, 5, 20))

                start += n_details
                
                print(details.shape)
                plt.hist(details.flatten(), bins=bins, density=True, alpha=0.2, label=f'Detail {ns}')

            xs = np.linspace(-5, 5, 1000)
            ys = norm.pdf(xs)
            plt.plot(xs, ys, label='Standard normal')

            plt.legend()
            plt.show()

    train_data = named_dataset(normalise_y(y_tr), transform_t(t_tr))
    val_data = named_dataset(normalise_y(y_val), transform_t(t_val))

# nle = MAFNeuralLikelihood(ndim, n_layers=5)

# nuking the nle with large n_layers and reduction_factor close to 1
# reduces evidence tension
# not sure, the accuracy of the evidence is quite noisy at nside=8
    nle = MAFSurjectiveNeuralLikelihood(
        ndim, 
        # n_layers=8, 
        # # data_reduction_factor=0.6, # lower implies larger reduction
        decoder_distribution='gaussian',
        conditioner='made',
        blocks=steps,
        maf_stack_size=15,
        conditioner_n_neurons=128,
        decoder_n_layers=3
    )
# low learning rate good, possibly training idiosyncrasies
    nle.train(hk.PRNGSequence(2), train_data, val_data, learning_rate=1e-5)
    nle.plot_loss_curve()

# OOM with students_t
    samples = nle.sample_likelihood_func(random.PRNGKey(2), theta0=transform_t(theta0))
    true_mean_likelihood = dipole.generate_dipole(
        *np.asarray(theta0[0, :])[:, None],
        make_poisson_draws=False
    ).squeeze()
    mean_samples = unnormalise_y(samples).mean(axis=0)

    hp.projview(
        mean_samples, nest=True, graticule=True, sub=211,
        title='NLE mean P(D | theta)'
    )
    hp.projview(
        true_mean_likelihood, nest=True, graticule=True, sub=212, 
        title='True mean P(D | theta)'
    )
    plt.show()

    evidence_diff = []
    sigmas = []
    for _ in range(N_EVIDENCE):
        x0 = dipole.generate_dipole(
            *np.asarray(
                theta0[0, :]
            ).reshape(4, 1)
        )
        data = normalise_y(x0)
        _, logdetjac = tform.cycle_healpix_tree(jnp.asarray(x0), last_nside=last_nside)
        def lnlike_jax(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
            theta: jnp.ndarray = transform_theta_jax(params)

            log_like = nle.evaluate_lnlike(theta[None, :], data)
            # total_log_like = (log_like + log_det_jac(data).sum(axis=-1)).squeeze()
            total_log_like = (log_like + logdetjac).squeeze()

            return total_log_like

        jax_ns = JaxNestedSampler(lnlike_jax, jax_prior)
        jax_ns.setup(rng_key, n_live=500, n_delete=50)
        nested_samples = jax_ns.run()

        classic_model = DipolePoisson(prior, nside=NSIDE, mask_map=mask_map)
        classic_inferer = LikelihoodBasedInferer(x0.squeeze(), classic_model)
        classic_inferer.run_ultranest()

        print(
            f"NLE Log Evidence: {nested_samples.logZ():.2f} "
            f"± {nested_samples.logZ(100).std():.2f}" # type: ignore
        )
        print(
            f"Classic Log Evidence: {classic_inferer.log_bayesian_evidence:.2f} "
            f"± {classic_inferer.log_bayesian_evidence_err:.2f}" # type: ignore
        )
        diff = nested_samples.logZ() - classic_inferer.log_bayesian_evidence # type: ignore
        sigma = (
            np.abs(diff) # type: ignore
          / np.sqrt(nested_samples.logZ(1000).std()**2 + classic_inferer.log_bayesian_evidence_err**2)) # type: ignore
        print(f'Tension: {sigma}')
        evidence_diff.append(diff)
        sigmas.append(sigma)
