import os
from typing import Callable, Optional
from anesthetic import NestedSamples
from blackjax.types import PRNGKey
from getdist import MCSamples, plots
import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt
import healpy as hp
from scipy.stats import norm
from dipolesbi.tools.configs import MultiRoundInfererConfig, SurjectiveNLEConfig, TrainingConfig
from dipolesbi.tools.dataloader import split_train_val, split_train_val_dict
from dipolesbi.tools.healpix_helpers import build_funnel_steps
from dipolesbi.tools.inference import JaxNestedSampler, LikelihoodBasedInferer
from dipolesbi.tools.maps import SimpleDipoleMap, SimpleDipoleMapJax
from surjectors.util import named_dataset
from dipolesbi.tools.models import DipolePoisson
from dipolesbi.tools.neural_flows import MAFSurjectiveNeuralLikelihood
from dipolesbi.tools.priors import DipolePrior
from dipolesbi.tools.priors_jax import DipolePriorJax
from dipolesbi.tools.simulator import Simulator
from dipolesbi.tools.transforms import BlankTransform, HaarWaveletTransform
from dipolesbi.tools.utils import jax_cart2sph, jax_sph2cart, load_dict_npz, save_dict_npz
from jax import lax
import datetime


class MultiRoundInferer:
    def __init__(
            self, 
            rng_key: PRNGKey,
            initial_proposal: DipolePriorJax,
            simulator_function: Callable[[PRNGKey, dict[str, jnp.ndarray], bool], jnp.ndarray],
            reference_observation: jnp.ndarray,
            multi_round_config: MultiRoundInfererConfig,
            nle_config: SurjectiveNLEConfig = SurjectiveNLEConfig.standard(),
            train_config: TrainingConfig = TrainingConfig()
    ) -> None:
        self.mr_config = multi_round_config

        self.rng_key = rng_key
        self.initial_proposal = initial_proposal
        self.initial_proposal.change_kwarg('N', 'mean_density')
        self.simulator_function = simulator_function
        self.data_ndim = reference_observation.shape[-1]
        self.nside = hp.npix2nside(self.data_ndim)
        self.reference_observation = reference_observation[None, :]
        self.sample_posterior_seed = 0

        if not os.path.exists(self.mr_config.plot_save_dir):
            os.makedirs(self.mr_config.plot_save_dir)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        new_plot_dir = os.path.join(self.mr_config.plot_save_dir, timestamp)
        os.makedirs(new_plot_dir, exist_ok=True)
        self.mr_config.plot_save_dir = new_plot_dir

        np.random.seed(42) # hopefully to keep ultranest results deterministic

        if self.mr_config.dequantise_data:
            self.ns_dequantise_key = jax.random.PRNGKey(0)
        
        self.nle_config = nle_config
        self.train_config = train_config

        self.theta_mean = None
        self.theta_std = None

        if self.mr_config.custom_data_transform is None:
            self.mr_config.custom_data_transform = BlankTransform()

        self.all_data = jnp.full((self.mr_config.simulation_budget, self.data_ndim), jnp.nan)
        self.all_theta = {
            'mean_density': jnp.full((self.mr_config.simulation_budget,), jnp.nan),
            'observer_speed': jnp.full((self.mr_config.simulation_budget,), jnp.nan),
            'dipole_longitude': jnp.full((self.mr_config.simulation_budget,), jnp.nan),
            'dipole_latitude': jnp.full((self.mr_config.simulation_budget), jnp.nan)
        }
        self.lnZ_per_round = []
        self.lnZerr_per_round = []
        self.final_nested_samples = None

    def run_preloaded(self) -> None:
        print(f'Loading simulations from {self.mr_config.simulation_path}...')
        self.load_simulations(self.mr_config.simulation_path)

        current_key = self.rng_key
        current_key, train_key = jax.random.split(current_key)
        self.current_round = self.mr_config.n_rounds

        self.trn_set, self.val_set = self._make_train_val_set()

        self.nle = self._instantiate_nle()

        print('Starting training...')
        self.nle.train(
            hk.PRNGSequence(train_key),
            self.trn_set, 
            self.val_set,
            config=self.train_config
        )
        self.nle.plot_loss_curve(
            show=False, 
            save_path=(
                self.mr_config.plot_save_dir
              + f'/loss_curve_{self.current_round}.png'
            )
        )
        self._inspect_learned_likelihood(train_key)

        self._compute_posterior()

        self._clear_data_summary_stats()

        self._benchmark_classic(current_key)

    def run(self):
        current_key = self.rng_key
        inspect_every = 1
        inspect_count = 0

        for round_idx in range(self.mr_config.n_rounds):
            self.current_round = round_idx

            current_key, proposal_key, sim_key, train_key = jax.random.split(
                current_key, 4
            )
            theta = self._sample_proposal(
                key=proposal_key,
                n_samples=self.mr_config.simulations_per_round,
                use_initial=True if round_idx == 0 else False
            )

            print('Generating simulations...')
            data = self._generate_simulations(sim_key, theta)
            self._add_to_simulation_pool(sim_key, data, theta)
            self.trn_set, self.val_set = self._make_train_val_set()
            del data; del theta

            self.nle = self._instantiate_nle()

            print('Starting training...')
            self.nle.train(
                hk.PRNGSequence(train_key),
                self.trn_set, 
                self.val_set,
                config=self.train_config
            )

            inspect_count += 1
            if (inspect_count == inspect_every):
                self.nle.plot_loss_curve(
                    show=False, 
                    save_path=(
                        self.mr_config.plot_save_dir
                      + f'/loss_curve_{self.current_round}.png'
                    )
                )
                self._inspect_learned_likelihood(train_key)
                inspect_count = 0

            self._compute_posterior()

            self._clear_data_summary_stats()

        self._benchmark_classic(current_key)
        self.final_nle_samples = self.nested_samples
        self.final_classic_samples = self.classic_nested_samples
        plt.close('all')

        self._dump_configs()

    def save_simulations(self) -> None:
        data_path = self.mr_config.plot_save_dir + '/data'
        os.mkdir(data_path)
        np.save(f'{data_path}/data.npy', jax.device_get(self.all_data))
        save_dict_npz(f'{data_path}/theta', self.all_theta)

    def load_simulations(self, simulation_path: str) -> None:
        self.all_data = jnp.array(np.load(f'{simulation_path}/data.npy', allow_pickle=True))
        self.all_theta = load_dict_npz(f'{simulation_path}/theta.npz')

    def _dump_configs(self) -> None:
        config_path = os.path.join(self.mr_config.plot_save_dir, "configs.txt")
        with open(config_path, "w") as f:
            f.write("MultiRoundInfererConfig:\n")
            f.write(str(self.mr_config) + "\n\n")
            f.write("SurjectiveNLEConfig:\n")
            f.write(str(self.nle_config) + "\n\n")
            f.write("NLE Instance:\n")
            f.write(str(self.nle) + "\n\n")
            f.write("TrainingConfig:\n")
            f.write(str(self.train_config) + "\n\n")
            f.write("TransformConfig:\n")
            f.write(str(self.mr_config.custom_data_transform) + "\n")

    def _inspect_learned_likelihood(self, rng_key: PRNGKey, n_repeats: int = 50_000) -> None:
        sample_key, simulate_key = jax.random.split(rng_key)

        assert self.mr_config.reference_theta is not None

        samples = self.nle.sample_likelihood_func(
            sample_key,
            theta0=jnp.repeat(
                self._transform_theta(self.mr_config.reference_theta, in_ns=True)[None, :],
                repeats=n_repeats,
                axis=0
            )
        )
        true_mean_likelihood = self.simulator_function(
            simulate_key,
            self.mr_config.reference_theta,
            False
        )
        self.mean_samples = self._untransform_data(samples).mean(axis=0)

        plt.figure()
        hp.projview(
            self.mean_samples, nest=True, graticule=True, sub=211,
            title='NLE mean P(D | theta)'
        )
        hp.projview(
            true_mean_likelihood, nest=True, graticule=True, sub=212, 
            title='True mean P(D | theta)'
        )
        plt.savefig(
            self.mr_config.plot_save_dir
          + f'/likelihood_{self.current_round}.png',
            bbox_inches='tight'
        )

    def _benchmark_classic(self, rng_key: PRNGKey) -> None:
        classic_model = SimpleDipoleMapJax(
            self.nside,
            reference_data=self.reference_observation
        )

        def lnlike_jax(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
            return classic_model.log_likelihood(params)

        self.classic_jax_ns = JaxNestedSampler(lnlike_jax, self.initial_proposal)
        self.classic_jax_ns.setup(rng_key, n_live=1000, n_delete=200)
        self.classic_nested_samples = self.classic_jax_ns.run()

        self.true_lnZ = self.classic_nested_samples.logZ()
        self.true_lnZerr = self.classic_nested_samples.logZ(100).std() # type: ignore
        print(
            f"NLE Log Evidence: {self.nested_samples.logZ():.2f} "
            f"± {self.nested_samples.logZ(100).std():.2f}" # type: ignore
        )
        print(
            f"Classic Log Evidence: {self.true_lnZ:.2f} "
            f"± {self.true_lnZerr:.2f}" # type: ignore
        )
        diff = self.nested_samples.logZ() - self.true_lnZ # type: ignore
        sigma = (
            np.abs(diff) # type: ignore
          / np.sqrt(
                self.nested_samples.logZ(100).std()**2  # type: ignore
              + self.true_lnZerr**2 # type: ignore
            ) 
        )
        print(f'Tension: {sigma}')

        nle_raw_samples = self.nested_samples.to_numpy()[:, :-3]
        idx_weights = self.nested_samples.index.to_numpy()
        weights = np.asarray([el[1] for el in idx_weights])

        classic_raw_samples = self.classic_nested_samples.to_numpy()[:, :-3]
        classic_idx_weights = self.classic_nested_samples.index.to_numpy()
        classic_weights = np.asarray([el[1] for el in classic_idx_weights])

        nle_samples = MCSamples(
            samples=nle_raw_samples,
            weights=weights,
            sampler='nested',
            names=self.initial_proposal.prior_names,
            labels=self.initial_proposal.prior_names
        )
        classic_samples = MCSamples(
            samples=classic_raw_samples, # type: ignore
            weights=classic_weights,
            sampler='nested',
            names=self.initial_proposal.prior_names,
            labels=self.initial_proposal.prior_names
        )
        g = plots.get_subplot_plotter()
        g.triangle_plot(
            [nle_samples, classic_samples],
            filled=True,
            markers=list(self.mr_config.reference_theta.values()), # type: ignore
            marker_args={'lw': 1}, # type: ignore
            legend_labels=['NLE', 'Truth']
        )
        plt.savefig(
            self.mr_config.plot_save_dir
          + f'/corner_final.png',
            bbox_inches='tight'
        )

    def _add_to_simulation_pool(
            self,
            rng_key: PRNGKey,
            data: jnp.ndarray,
            theta: dict[str, jnp.ndarray]
    ) -> None:
        start = self.current_round * self.mr_config.simulations_per_round
        end = (self.current_round + 1) * self.mr_config.simulations_per_round

        if self.mr_config.dequantise_data:
            _, dequantise_key = jax.random.split(rng_key)
            data += jax.random.uniform(
                dequantise_key, 
                shape=data.shape, 
                minval=-0.5, 
                maxval=0.5
            )

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

        # if the data has been dequantised, we need to marginalise out
        # over the introduced noise; this amounts to averaging permuted
        # x0s, but it has to be deterministic for the NS lnlike call
        if self.mr_config.dequantise_data:
            _, self.ns_dequantise_key = jax.random.split(self.ns_dequantise_key)
            epsilon = jax.random.uniform(
                self.ns_dequantise_key,
                shape=(self.mr_config.n_requantisations, self.data_ndim), # type: ignore
                minval=-0.5, 
                maxval=0.5
            )
            x0 = self.reference_observation + epsilon
        else:
            x0 = self.reference_observation

        z0, log_det_jac = self._transform_data(x0)

        # log_det_jac will be of length n_requantisations, and since we
        # only add a scalar to each dim, the log_det_jac is the same for
        # each dequantisation
        if self.mr_config.dequantise_data:
            log_det_jac = log_det_jac[0]

        def lnlike_jax(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
            theta = self._transform_theta(params, in_ns=True)

            if not self.mr_config.dequantise_data:
                log_like = self.nle.evaluate_lnlike(theta[None, :], z0)
            else:
                log_like_by_permbatch = jax.vmap(
                    lambda zi: self.nle.evaluate_lnlike(theta[None, :], zi[None, :])
                )(z0)
                log_like = (
                    jax.scipy.special.logsumexp(log_like_by_permbatch, axis=0)
                  - jnp.log(self.mr_config.n_requantisations) # type: ignore
                )

            log_like += log_det_jac
            return log_like.squeeze()

        self.jax_ns = JaxNestedSampler(lnlike_jax, self.initial_proposal)
        self.jax_ns.setup(ns_key, n_live=1000, n_delete=200)
        self.nested_samples = self.jax_ns.run()

        kinds = {'lower': 'kde_2d', 'diagonal': 'hist_1d', 'upper': 'scatter_2d'}
        plt.figure()
        self.nested_samples.plot_2d(
            self.initial_proposal.simulator_kwargs,
            kinds=kinds
        )
        plt.savefig(
            self.mr_config.plot_save_dir
          + f'/jax_samples_{self.current_round}.png',
            bbox_inches='tight'
        )

        self.lnZ_per_round.append(self.nested_samples.logZ())
        self.lnZerr_per_round.append(self.nested_samples.logZ(100).std()) # type: ignore

    def _make_train_val_set(self) -> tuple[named_dataset, named_dataset]:
        (trn_data, val_data), (trn_theta, val_theta) = self._split_train_val(
            self.all_data, self.all_theta
        )

        self._compute_theta_norm(trn_theta)

        norm_train_data, _ = self._transform_data(trn_data)
        norm_val_data, _ = self._transform_data(val_data)
        norm_train_theta = self._transform_theta(trn_theta)
        norm_val_theta = self._transform_theta(val_theta)

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
        cur_max_idx = (self.current_round+1) * self.mr_config.simulations_per_round
        cur_data = data[:cur_max_idx, :]
        cur_theta = jax.tree_map(lambda x: x[:cur_max_idx], theta)
        return split_train_val_dict(cur_data, cur_theta)

    def _compute_theta_norm(
            self,
            train_theta: dict[str, jnp.ndarray]
    ) -> None:
        mean_nbar = jnp.nanmean(train_theta['mean_density'])
        std_nbar = jnp.nanstd(train_theta['mean_density'])
        mean_v = jnp.nanmean(train_theta['observer_speed'])
        std_v = jnp.nanstd(train_theta['observer_speed'])
        self.theta_mean = jnp.asarray([mean_nbar, mean_v, 0, 0])
        self.theta_std = jnp.asarray([std_nbar, std_v, 1, 1])

    def _transform_theta(
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

    def _transform_data(self, data: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Normalise only using stats computed from the training data.
        '''
        return self.mr_config.custom_data_transform(data)

    def _untransform_data(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.mr_config.custom_data_transform.inverse(z)

    def _clear_data_summary_stats(self) -> None:
        self.mr_config.custom_data_transform.clear()

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
            n_posterior = self.mr_config.simulations_per_round - n_prior
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
        sample_keys = jax.random.split(key, self.mr_config.simulations_per_round)
        simulations = jax.vmap(self.simulator_function)(sample_keys, theta)
        return simulations

    def _instantiate_nle(self) -> MAFSurjectiveNeuralLikelihood:
        nle = MAFSurjectiveNeuralLikelihood(
            self.data_ndim,
            config=self.nle_config
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
        z, logdet = tform._cycle_healpix_tree(y, last_nside=last_nside)
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
        _, logdetjac = tform._cycle_healpix_tree(jnp.asarray(x0), last_nside=last_nside)
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
