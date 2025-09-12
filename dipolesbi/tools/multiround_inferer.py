import os
import time
from typing import Callable, Optional
from anesthetic import NestedSamples
from blackjax.types import PRNGKey
from getdist import MCSamples, plots
import haiku as hk
import jax
import numpy as np
from numpy.typing import NDArray
from jax import numpy as jnp
from matplotlib import pyplot as plt
import healpy as hp
from dipolesbi.tools.configs import MultiRoundInfererConfig, SurjectiveNLEConfig, TrainingConfig, TransformConfig
from dipolesbi.tools.dataloader import split_train_val_dict
from dipolesbi.tools.inference import JaxNestedSampler
from dipolesbi.tools.maps import SimpleDipoleMapJax
from surjectors.util import named_dataset
from dipolesbi.tools.dataloader import named_dataset_idx
from dipolesbi.tools.neural_flows import MAFSurjectiveNeuralLikelihood
from dipolesbi.tools.np_rngkey import NPKey, npkey_from_jax
from dipolesbi.tools.priors_np import DipolePriorNP
from dipolesbi.tools.transforms import BlankTransform
from dipolesbi.tools.ui import MultiRoundInfererUI
from dipolesbi.tools.utils import HidePrints, jax_sph2cart, load_dict_npz, np_sph2cart, save_dict_npz
from jax import lax
import datetime


class MultiRoundInferer:
    def __init__(
            self, 
            initial_proposal: DipolePriorNP,
            simulator_function: Callable[
                [NPKey, dict[str, NDArray[np.float32]], bool],
                NDArray[np.float32]
            ],
            reference_observation: NDArray,
            multi_round_config: MultiRoundInfererConfig,
            transform_config: Optional[TransformConfig] = None,
            nle_config: SurjectiveNLEConfig = SurjectiveNLEConfig.standard(),
            train_config: TrainingConfig = TrainingConfig(),
    ) -> None:
        self.mr_config = multi_round_config

        self.rng_key = jax.random.PRNGKey(self.mr_config.prng_integer_seed)

        self.initial_proposal = initial_proposal
        self.initial_proposal.change_kwarg('N', 'mean_density')
        self.initial_proposal_jax = initial_proposal.to_jax()
        self.initial_proposal_jax.change_kwarg('N', 'mean_density')

        self.simulator_function = simulator_function
        self.data_ndim = reference_observation.shape[-1]
        self.nside = hp.npix2nside(self.data_ndim)
        self.reference_observation = reference_observation
        self.sample_posterior_seed = 0

        if not os.path.exists(self.mr_config.plot_save_dir):
            os.makedirs(self.mr_config.plot_save_dir)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamp_and_seed = f'{timestamp}_SEED{self.mr_config.prng_integer_seed}'
        new_plot_dir = os.path.join(self.mr_config.plot_save_dir, timestamp_and_seed)
        os.makedirs(new_plot_dir, exist_ok=True)
        self.mr_config.plot_save_dir = new_plot_dir

        self.nle_config = nle_config
        self.train_config = train_config
        self.nle = None

        self.theta_mean = None
        self.theta_std = None

        if transform_config is None:
            self.data_transform = BlankTransform()
            self.transform_config = None
        else:
            self.data_transform = transform_config.transform
            self.transform_config = transform_config

        self.all_data = np.full(
            (self.mr_config.simulation_budget, self.data_ndim), np.nan,
            dtype=np.float32
        )
        self.all_theta = {
            k: np.full((self.mr_config.simulation_budget,), np.nan, dtype=np.float32)
            for k in self.initial_proposal.simulator_kwargs
        }
        self.round_idx_badval = -1
        self.all_round_id = np.full(
            self.mr_config.simulation_budget, 
            self.round_idx_badval, # bad (unfilled) int value
            dtype=np.int32
        )

        self.lnZ_per_round = []
        self.lnZerr_per_round = []
        self.final_nested_samples = None

    # def run_preloaded(self) -> None:
    #     print(f'Loading simulations from {self.mr_config.simulation_path}...')
    #     self.load_simulations(self.mr_config.simulation_path)
    #
    #     current_key = self.rng_key
    #     current_key, train_key = jax.random.split(current_key)
    #     self.current_round = self.mr_config.n_rounds
    #
    #     self.trn_set, self.val_set = self._make_train_val_set()
    #
    #     self.nle = self._instantiate_nle()
    #
    #     print('Starting training...')
    #     self.nle.train(
    #         hk.PRNGSequence(train_key),
    #         self.trn_set, 
    #         self.val_set,
    #         config=self.train_config
    #     )
    #     self.nle.plot_loss_curve(
    #         show=False, 
    #         save_path=(
    #             self.mr_config.plot_save_dir
    #           + f'/loss_curve_{self.current_round}.png'
    #         )
    #     )
    #     self._inspect_learned_likelihood(train_key)
    #
    #     self._compute_posterior()
    #
    #     self._clear_data_summary_stats()
    #
    #     self._benchmark_classic(current_key)

    def run(self):
        current_key = self.rng_key
        tasks = [
            'Sample proposal', 'Generate simulations',
            'Train NLE', 'Sample likelihood', 'Compute posterior',
            'Benchmark'
        ]
        self.ui = MultiRoundInfererUI(tasks)

        with self.ui.session(refresh_per_second=20):

            time.sleep(1) # avoid spam
            self.ui.begin_global_progress(total=self.mr_config.simulation_budget)
            self.ui.set_stats_columns(columns=['Round', 'Evidence'])

            for round_idx in range(self.mr_config.n_rounds):
                self.ui.reset()
                self.ui.set_round(round_idx, self.mr_config.n_rounds)
                self.current_round = round_idx
                self.ui.add_stats_row(row={'Round': round_idx+1, 'Evidence': ""})

                npkey = npkey_from_jax(current_key)
                proposal_key, sim_key, split_key = npkey.split(3)
                current_key, train_key, inspect_key, posterior_key = jax.random.split(
                    current_key, 4
                )

                self.ui.start_step(0, subtitle='sampling')
                theta = self._sample_proposal(
                    key=proposal_key,
                    n_samples=self.mr_config.simulations_per_round,
                    use_initial=True if round_idx == 0 else False
                )
                self.ui.finish_step('sampled')

                self.ui.start_step(1, subtitle='simulating')
                data = self._generate_simulations(sim_key, theta)
                self._add_to_simulation_pool(sim_key, data, theta)
                self.trn_set, self.val_set = self._make_train_val_set(split_key)
                del data; del theta
                self.ui.finish_step('simulated')

                self.ui.start_step(2, subtitle='training')
                self.nle = self._instantiate_nle()
                self._train_nle(train_key)
                self.ui.finish_step('trained')

                if round_idx == 0:
                    self._dump_configs() # dump after training

                n_samps = self.mr_config.n_likelihood_samples
                self.ui.start_step(3, 'inspecting', total=n_samps)
                self._inspect_learned_likelihood(inspect_key, n_samps)
                self.ui.finish_step('inspected')

                self.ui.start_step(4, 'computing')
                self._compute_posterior(posterior_key)
                self.ui.update_last_stats_row({'Evidence': self.jax_ns.evidence_str})
                self.ui.finish_step('computed')

                self._clear_data_summary_stats()
                plt.close('all')
                self.ui.advance_global(n=self.mr_config.simulations_per_round)

        self.ui.start_step(5, 'benchmarking')
        self._benchmark_classic(current_key)
        self.final_nle_samples = self.nested_samples
        self.final_classic_samples = self.classic_nested_samples
        self.ui.finish_step('benchmarked')

    def _train_nle(self, train_key: PRNGKey) -> None:
        assert self.nle is not None

        self.nle.train(
            hk.PRNGSequence(train_key),
            self.trn_set, 
            self.val_set,
            config=self.train_config,
            ui=self.ui
        )
        self.nle.plot_loss_curve(
            show=False, 
            save_path=(
                self.mr_config.plot_save_dir
              + f'/loss_curve_{self.current_round}.png'
            )
        )

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
            f.write(str(self.transform_config) + "\n")

    def _sample_likelihood_stream(
            self,
            rng_key: PRNGKey,
            n_repeats: int,
            chunk_bytes_gb: float = 0.25
    ) -> NDArray[np.float32]:
        assert self.mr_config.reference_theta is not None
        assert self.nle is not None

        ndim = self.data_ndim
        bytes_per_elem = 4  # float32

        # Heuristic chunk size: keep output under chunk_bytes_gb on device
        max_per_chunk = max(
            1, 
            int((chunk_bytes_gb * (1024**3)) // (ndim * bytes_per_elem))
        )
        m = min(n_repeats, max_per_chunk)
        self.ui.begin_progress(total=n_repeats)

        out = np.empty((n_repeats, ndim), dtype=np.float32)

        theta0 = self._transform_theta(self.mr_config.reference_theta)
        theta0_ndim = theta0.shape[-1]

        start = 0
        while start < n_repeats:
            end = min(n_repeats, start + m)
            batch_size = end - start

            theta_chunk = jnp.broadcast_to(theta0, (batch_size, theta0_ndim))

            rng_key, subkey = jax.random.split(rng_key)

            samples = self.nle.sample_likelihood_func(
                subkey,
                theta0=theta_chunk,
                sample_shape=(batch_size,)
            )

            samples_host = jax.device_get(samples)
            out[start:end] = samples
            del samples, samples_host, theta_chunk

            start = end
            self.ui.update_progress(batch_size)
        self.ui.end_progress()

        return out

    def _inspect_learned_likelihood(self, rng_key: PRNGKey, n_repeats: int = 50_000) -> None:
        _, simulate_key = jax.random.split(rng_key)

        assert self.mr_config.reference_theta is not None
        assert self.nle is not None

        samples = self._sample_likelihood_stream(rng_key, n_repeats)
        true_mean_likelihood = self.simulator_function(
            simulate_key,
            self.mr_config.reference_theta,
            False
        ).squeeze()

        samples = jax.device_get(samples)
        samples_untransformed = self._untransform_data(samples)
        self.mean_samples = samples_untransformed.mean(axis=0)

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
            reference_data=jnp.asarray(self.reference_observation)
        )

        def lnlike_jax(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
            return classic_model.log_likelihood(params)

        self.classic_jax_ns = JaxNestedSampler(
            lnlike_jax, 
            self.initial_proposal_jax,
            ui=self.ui
        )
        self.classic_jax_ns.setup(rng_key, n_live=1000, n_delete=200)
        self.classic_nested_samples = self.classic_jax_ns.run()

        self.true_lnZ = self.classic_nested_samples.logZ()
        self.true_lnZerr = self.classic_nested_samples.logZ(100).std() # type: ignore
        self.ui.log(
            f"NLE Log Evidence: {self.nested_samples.logZ():.2f} "
            f"± {self.nested_samples.logZ(100).std():.2f}" # type: ignore
        )
        self.ui.log(
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
        self.ui.log(f'Tension: {sigma}')

        nle_raw_samples = self.nested_samples.to_numpy()[:, :-3]
        idx_weights = self.nested_samples.index.to_numpy()
        weights = np.asarray([el[1] for el in idx_weights])

        classic_raw_samples = self.classic_nested_samples.to_numpy()[:, :-3]
        classic_idx_weights = self.classic_nested_samples.index.to_numpy()
        classic_weights = np.asarray([el[1] for el in classic_idx_weights])

        with HidePrints():
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
            rng_key: NPKey,
            data: NDArray,
            theta: dict[str, NDArray]
    ) -> None:
        start = self.current_round * self.mr_config.simulations_per_round
        end = (self.current_round + 1) * self.mr_config.simulations_per_round

        if self.mr_config.dequantise_data:
            _, dequantise_key = rng_key.split(2)
            data += dequantise_key.uniform(shape=data.shape, low=-0.5, high=0.5)

        self.ui.log(f'Adding data from idx {start} to idx {end}...')
        self.all_data[start:end, :] = data
        self.all_round_id[start:end] = self.current_round

        for key in self.all_theta.keys():
            self.all_theta[key][start:end] = theta[key]

    def _write_contiguous(self, storage, batch, start: int):
        def put(dst, src):
            # write src into dst at [start:start+src.shape[0]] along axis 0
            idx = (start,) + (0,) * (dst.ndim - 1)
            return lax.dynamic_update_slice(dst, src, idx)
        return jax.tree.map(put, storage, batch)

    def _compute_posterior(self, posterior_key: PRNGKey) -> None:
        ns_key, dequantise_key = jax.random.split(posterior_key)

        # if the data has been dequantised, we need to marginalise out
        # over the introduced noise; this amounts to averaging permuted
        # x0s, but it has to be deterministic for the NS lnlike call
        if self.mr_config.dequantise_data:
            epsilon = jax.random.uniform(
                dequantise_key,
                shape=(self.mr_config.n_requantisations, self.data_ndim), # type: ignore
                minval=-0.5, 
                maxval=0.5
            )
            x0 = self.reference_observation + epsilon
        else:
            x0 = self.reference_observation

            # add batch dimension if none provided
            if x0.ndim == 1:
                x0 = x0[None, :]

        z0, log_det_jac = self._transform_data(np.asarray(x0))
        z0 = jax.device_put(z0)
        log_det_jac = jax.device_put(log_det_jac)

        # log_det_jac will be of length n_requantisations, and since we
        # only add a scalar to each dim, the log_det_jac is the same for
        # each dequantisation
        if self.mr_config.dequantise_data:
            log_det_jac = log_det_jac[0]

        def lnlike_jax(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
            theta = self._transform_theta_jax(params, in_ns=True)

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

        self.jax_ns = JaxNestedSampler(
            lnlike_jax, 
            self.initial_proposal_jax, 
            self.ui
        )
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

    def _make_train_val_set(self, split_key: NPKey) -> tuple[named_dataset_idx, named_dataset]:
        (trn_data, val_data), (trn_theta, val_theta), rnd_idx = self._split_train_val(
            self.all_data, # type: ignore
            self.all_theta, 
            self.all_round_id,
            split_key=split_key
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
        assert not jnp.any(rnd_idx == self.round_idx_badval)

        trn_set = named_dataset_idx(norm_train_data, norm_train_theta, rnd_idx)
        val_set = named_dataset(norm_val_data, norm_val_theta)

        return trn_set, val_set

    def _split_train_val(
            self, 
            data: NDArray, 
            theta: dict[str, NDArray],
            round_idx: NDArray,
            split_key: NPKey
    ) -> tuple[
        tuple[NDArray, NDArray], 
        tuple[dict[str, NDArray], dict[str, NDArray]],
        NDArray
    ]:
        cur_max_idx = (self.current_round+1) * self.mr_config.simulations_per_round
        cur_data = data[:cur_max_idx, :]
        cur_theta = {key: theta[key][:cur_max_idx] for key in theta.keys()}
        cur_round_idx = round_idx[:cur_max_idx]
        return split_train_val_dict(
            key=split_key,
            y=cur_data,
            x=cur_theta,
            round_idx=cur_round_idx,
            validation_fraction=self.train_config.validation_fraction
        )

    def _compute_theta_norm(
            self,
            train_theta: dict[str, NDArray]
    ) -> None:
        mean_nbar = np.nanmean(train_theta['mean_density'])
        std_nbar = np.nanstd(train_theta['mean_density'])
        mean_v = np.nanmean(train_theta['observer_speed'])
        std_v = np.nanstd(train_theta['observer_speed'])
        self.theta_mean = np.asarray([mean_nbar, mean_v, 0, 0])
        self.theta_std = np.asarray([std_nbar, std_v, 1, 1])

    def _transform_theta_jax(
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

    def _transform_theta(
        self,
        theta: dict[str, NDArray]
    ) -> NDArray:
        lon = np.deg2rad(theta['dipole_longitude'])
        colat = np.pi / 2 - np.deg2rad(theta['dipole_latitude'])
        x, y, z = np_sph2cart(lon, colat)

        assert self.theta_mean is not None
        assert self.theta_std is not None

        t_transformed = np.stack(
            [
                (theta['mean_density'] - self.theta_mean[0]) / self.theta_std[0],
                (theta['observer_speed'] - self.theta_mean[1]) / self.theta_std[1],
                x, y, z
            ],
            axis=1
        )
        return t_transformed

    def _transform_data(self, data: NDArray) -> tuple[NDArray, NDArray]:
        '''
        Normalise only using stats computed from the training data.
        '''
        return self.data_transform(data) # type: ignore

    def _untransform_data(self, z: np.ndarray) -> np.ndarray:
        return self.data_transform.inverse(z) # type: ignore

    def _clear_data_summary_stats(self) -> None:
        self.data_transform.clear() # type: ignore

    def _sample_proposal(
            self,
            key: NPKey,
            n_samples: int, 
            use_initial: bool = True
    ) -> dict[str, NDArray]:
        if use_initial:
            self.ui.log('Sampling from initial proposal...')
            prior_samples = self.initial_proposal.sample(key, n_samples)
        else:
            n_prior = int(self.mr_config.initial_fraction * n_samples)
            n_posterior = n_samples - n_prior
            assert n_prior + n_posterior == n_samples
            self.ui.log(
                f'Sampling from learned posterior {1 - self.mr_config.initial_fraction} '
                f'and initial proposal {self.mr_config.initial_fraction}...'
            )

            if n_posterior > 0:
                self.sample_posterior_seed += 1
                posterior_samples = self.nested_samples.sample(
                    n=n_posterior,
                    random_state=self.sample_posterior_seed
                )
                posterior_samples = self._reformat_samples(posterior_samples)
            else:
                posterior_samples = None

            if n_prior > 0:
                _, init_key = key.split()
                initial_samples = self.initial_proposal.sample(init_key, n_prior)
            else:
                initial_samples = None

            if initial_samples and posterior_samples:
                prior_samples = {
                    k: np.concatenate([posterior_samples[k], initial_samples[k]], axis=0)
                    for k in posterior_samples.keys()
                }
            else:
                if initial_samples and not posterior_samples:
                    prior_samples = initial_samples
                elif posterior_samples and not initial_samples:
                    prior_samples = posterior_samples
                else:
                    raise Exception(
                        'Somehow no posterior or initial samples were generated.'
                    )
            
        return prior_samples

    def _reformat_samples(
            self,
            samples: NestedSamples
    ) -> dict[str, NDArray[np.float32]]:
        samples_dict = samples.to_dict(orient='list')

        del samples_dict['logL']
        del samples_dict['logL_birth']
        del samples_dict['nlive']

        return {
            key: np.asarray(val, dtype=np.float32) for key, val in samples_dict.items()
        }

    def _generate_simulations(
            self, 
            key: NPKey,
            theta: dict[str, NDArray]
    ) -> NDArray:
        return self.simulator_function(key, theta, True)

    def _instantiate_nle(self) -> MAFSurjectiveNeuralLikelihood:
        if (not self.train_config.restore_from_previous) or (self.nle is None):
            nle = MAFSurjectiveNeuralLikelihood(
                self.data_ndim,
                config=self.nle_config,
                data_transform=self.data_transform
            )
        else:
            nle = self.nle
        return nle
