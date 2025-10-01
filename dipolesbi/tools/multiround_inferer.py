import os
import time
from typing import Callable, Literal, Optional, cast
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
from dipolesbi.tools.configs import (
    MultiRoundInfererConfig, 
    NeuralFlowConfig, 
    TrainingConfig, 
    TransformConfig
)
from dipolesbi.tools.dataloader import split_train_val_dict
from dipolesbi.tools.inference import JaxNestedSampler
from dipolesbi.tools.dataloader import healpix_map_dataset, healpix_map_dataset_idx
from dipolesbi.tools.neural_flows import NeuralFlow
from dipolesbi.tools.np_rngkey import NPKey, npkey_from_jax
from dipolesbi.tools.priors_np import DipolePriorNP
from dipolesbi.tools.ui import MultiRoundInfererUI
from dipolesbi.tools.utils import HidePrints, load_dict_npz, save_dict_npz
from jax import lax
import datetime
from corner import corner


class MultiRoundInferer:
    def __init__(
            self, 
            mode: Literal['NLE'] | Literal['NPE'],
            initial_proposal: DipolePriorNP,
            simulator_function: Callable[
                [NPKey, dict[str, NDArray[np.float32]], bool],
                tuple[NDArray[np.float32], NDArray[np.bool_]]
            ],
            reference_observation: tuple[NDArray, NDArray[np.bool_]],
            multi_round_config: MultiRoundInfererConfig,
            nflow_config: NeuralFlowConfig,
            transform_config: TransformConfig,
            train_config: TrainingConfig = TrainingConfig(),
            true_logl: Optional[Callable[[dict[str, jnp.ndarray]], jnp.ndarray]] = None
    ) -> None:
        self.mode = mode
        self.mr_config = multi_round_config

        self.rng_key = jax.random.PRNGKey(self.mr_config.prng_integer_seed)

        self.initial_proposal = initial_proposal
        self.initial_proposal_jax = initial_proposal.to_jax()

        self.simulator_function = simulator_function
        self.reference_data = reference_observation[0]
        self.reference_mask = reference_observation[1]
        self.true_logl = true_logl
        self.data_ndim = self.reference_data.shape[-1]
        self.theta_ndim = initial_proposal.ndim
        self.nside = hp.npix2nside(self.data_ndim)

        reference_theta = self.mr_config.reference_theta
        if reference_theta is not None:
            self.reference_theta_jax: dict[str, jnp.ndarray] = {
                k: jnp.array(v) for k, v in reference_theta.items()
            }
        else:
            self.reference_theta_jax: dict[str, jnp.ndarray] = {}

        self.sample_posterior_seed = 0

        if not os.path.exists(self.mr_config.plot_save_dir):
            os.makedirs(self.mr_config.plot_save_dir)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamp_and_seed = f'{timestamp}_SEED{self.mr_config.prng_integer_seed}'
        new_plot_dir = os.path.join(self.mr_config.plot_save_dir, timestamp_and_seed)
        os.makedirs(new_plot_dir, exist_ok=True)
        self.mr_config.plot_save_dir = new_plot_dir

        self.nflow_config = nflow_config
        self.train_config = train_config
        self.nflow = None

        self.theta_mean = None
        self.theta_std = None

        self.data_transform = transform_config.data_transform_config.data_transform
        self.theta_transform = transform_config.theta_transform_config.theta_transform
        self.transform_config = transform_config
        self.embedding_net_config = self.transform_config.data_transform_config.embedding_net_config

        self.all_data = np.full(
            (self.mr_config.simulation_budget, self.data_ndim), np.nan,
            dtype=np.float32
        )
        self.all_mask = np.full(
            (self.mr_config.simulation_budget, self.data_ndim), 1,
            dtype=np.bool_
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
        self.classic_nested_samples = None

    @property
    def target_ndim(self) -> int:
        if self.mode == 'NLE':
            return self.data_ndim
        elif self.mode == 'NPE':
            return self.theta_ndim
        else:
            raise Exception(f'Mode {self.mode} not recognised.')

    def run(self) -> None:
        if self.mode == 'NLE':
            self._nle_pipeline()
        elif self.mode == 'NPE':
            self._npe_pipeline()

    def _nle_pipeline(self):
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
                self.nflow = self._instantiate_nflow()
                self._train_nflow(train_key)
                self.ui.finish_step('trained')

                if round_idx == 0:
                    self._dump_configs() # dump after training

                if self.mr_config.reference_theta is not None:
                    n_samps = self.mr_config.n_likelihood_samples
                    self.ui.start_step(3, 'inspecting', total=n_samps)
                    self._inspect_learned_likelihood(inspect_key, n_samps)
                    self.ui.finish_step('inspected')

                self.ui.start_step(4, 'computing')
                self._compute_posterior(posterior_key)
                self.ui.update_last_stats_row({'Evidence': self.jax_ns.evidence_str})
                self.ui.finish_step('computed')

                plt.close('all')
                self.ui.advance_global(n=self.mr_config.simulations_per_round)

        self.ui.start_step(5, 'benchmarking')
        if self.true_logl is not None: self._benchmark_classic(current_key)
        self.ui.finish_step('benchmarked')

        if self.mr_config.write_results_to_disk: self._write_results_to_disk()
        self._lnZ_plot()

    def _npe_pipeline(self):
        current_key = self.rng_key
        tasks = [
            'Sample proposal', 'Generate simulations',
            'Train NPE', 'Sample posterior'
        ]
        self.ui = MultiRoundInfererUI(tasks)

        with self.ui.session(refresh_per_second=20):

            time.sleep(1) # avoid spam
            self.ui.begin_global_progress(total=self.mr_config.simulation_budget)

            for round_idx in range(self.mr_config.n_rounds):
                self.ui.reset()
                self.ui.set_round(round_idx, self.mr_config.n_rounds)
                self.current_round = round_idx

                npkey = npkey_from_jax(current_key)
                proposal_key, sim_key, split_key = npkey.split(3)
                current_key, train_key, posterior_key = jax.random.split(
                    current_key, 3
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
                self.nflow = self._instantiate_nflow()
                self._train_nflow(train_key)
                self.ui.finish_step('trained')

                if round_idx == 0:
                    self._dump_configs() # dump after training

                self.ui.start_step(3, 'computing')
                self.posterior_samples = self._sample_posterior(
                    posterior_key, self.mr_config.n_posterior_samples
                )
                self.ui.finish_step('computed')

                plt.close('all')
                self.ui.advance_global(n=self.mr_config.simulations_per_round)

        self.ui.start_step(5, 'benchmarking')
        if self.true_logl is not None: self._benchmark_classic(current_key)
        self.ui.finish_step('benchmarked')

        if self.mr_config.write_results_to_disk: self._write_results_to_disk()

    def _train_nflow(self, train_key: PRNGKey) -> None:
        assert self.nflow is not None

        self.nflow.train(
            hk.PRNGSequence(train_key),
            self.trn_set, 
            self.val_set,
            config=self.train_config,
            ui=self.ui,
            prior=self.initial_proposal_jax if self.mode == 'NPE' else None,
        )
        self.nflow.plot_loss_curve(
            show=False, 
            save_path=(
                self.mr_config.plot_save_dir
              + f'/loss_curve_{self.current_round}.png'
            )
        )

    def _write_results_to_disk(self):
        if self.mode == 'NLE':
            self.final_posterior_samples = self.nested_samples
            self.final_posterior_samples.to_csv(
                path_or_buf=os.path.join(self.mr_config.plot_save_dir, 'samples.csv')
            )
        else:
            self.final_posterior_samples = self.posterior_samples
            np_samples = {
                k: np.array(v) for k, v in self.posterior_samples.items()
            }
            np.savez(
                os.path.join(self.mr_config.plot_save_dir, 'samples.npz'),
                **np_samples
            )

        self.final_classic_samples = self.classic_nested_samples

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
            f.write(str(self.nflow_config) + "\n\n")
            f.write("NLE Instance:\n")
            f.write(str(self.nflow) + "\n\n")
            f.write("TrainingConfig:\n")
            f.write(str(self.train_config) + "\n\n")
            f.write("TransformConfig:\n")
            f.write(str(self.transform_config) + "\n")

    def _to_jnp_array(self, theta: dict[str, jnp.ndarray]) -> jnp.ndarray:
        arrays = [theta[k] for k in theta.keys()]
        return jnp.stack(arrays, axis=1)

    def _sample_likelihood_stream(
            self,
            rng_key: PRNGKey,
            n_repeats: int,
            chunk_bytes_gb: float = 0.25
    ) -> NDArray[np.float32]:
        assert self.mr_config.reference_theta is not None
        assert self.nflow is not None

        ndim = self.data_ndim
        bytes_per_elem = np.dtype(np.float32).itemsize
        max_per_chunk = max(
            1,
            int((chunk_bytes_gb * (1024 ** 3)) // (ndim * bytes_per_elem))
        )
        chunk_size = min(n_repeats, max_per_chunk)
        self.ui.begin_progress(total=n_repeats)

        out = np.empty((n_repeats, ndim), dtype=np.float32)

        if self.theta_transform is not None:
            theta0, _ = self.theta_transform(self.reference_theta_jax)
        else:
            theta0 = self._to_jnp_array(self.reference_theta_jax)

        theta0_ndim = theta0.shape[-1]
        mask_jax = jax.device_put(self.reference_mask)

        start = 0
        while start < n_repeats:
            batch_size = min(n_repeats - start, chunk_size)
            end = start + batch_size

            theta_chunk = jnp.broadcast_to(theta0, (batch_size, theta0_ndim))

            rng_key, subkey = jax.random.split(rng_key)

            samples = self.nflow.sample_likelihood_func(
                subkey,
                theta0=theta_chunk,
                mask=mask_jax,
                sample_shape=(batch_size,)
            )

            samples_host = jax.device_get(samples).astype(np.float32, copy=False)
            out[start:end] = samples_host
            del samples, samples_host, theta_chunk

            start = end
            self.ui.update_progress(batch_size)
        self.ui.end_progress()

        return out

    def _inspect_learned_likelihood(self, rng_key: PRNGKey, n_repeats: int = 50_000) -> None:
        _, simulate_key = jax.random.split(rng_key)

        assert self.mr_config.reference_theta is not None
        assert self.nflow is not None

        samples = self._sample_likelihood_stream(rng_key, n_repeats) # (B, P)
        true_mean_likelihood, mask = self.simulator_function(
            simulate_key,
            self.mr_config.reference_theta,
            False
        )
        true_mean_likelihood = true_mean_likelihood.squeeze()

        samples = jax.device_get(samples)

        # this is a bit shit --- we need to get the transformed mask (zmask)
        assert self.nflow.data_transform is not None
        (_, zmask), _ = self.nflow.data_transform(
            jax.device_put(self.reference_data), jax.device_put(self.reference_mask)
        )
        zmask = zmask.repeat(repeats=samples.shape[0], axis=0)

        (samples_untransformed, mask), _ = self._untransform_data_and_logdet(
            samples, zmask
        )
        self.mean_samples = samples_untransformed.mean(axis=0)
        self.mean_samples[~mask[0, :]] = np.nan

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
          + f'/likelihood_{self.current_round}.pdf',
            bbox_inches='tight'
        )

    def _get_true_posterior_and_evidence(
            self, 
            rng_key: PRNGKey
    ) -> tuple[float, float, NestedSamples]:
        assert self.true_logl is not None

        self.classic_jax_ns = JaxNestedSampler(
            self.true_logl, 
            self.initial_proposal_jax,
            ui=self.ui
        )
        self.classic_jax_ns.setup(rng_key, n_live=1000, n_delete=200)
        self.classic_nested_samples = self.classic_jax_ns.run()

        self.true_lnZ = float(self.classic_nested_samples.logZ()) # type: ignore
        self.true_lnZerr = float(self.classic_nested_samples.logZ(100).std()) # type: ignore
        if self.mode == 'NLE':
            self.ui.log(
                f"NLE Log Evidence: {self.nested_samples.logZ():.2f} "
                f"± {self.nested_samples.logZ(100).std():.2f}" # type: ignore
            )
        self.ui.log(
            f"Classic Log Evidence: {self.true_lnZ:.2f} "
            f"± {self.true_lnZerr:.2f}" # type: ignore
        )
        return self.true_lnZ, self.true_lnZerr, self.classic_nested_samples

    def _benchmark_classic(self, rng_key: PRNGKey) -> None:
        self.true_lnZ, self.true_lnZerr, self.classic_nested_samples = (
            self._get_true_posterior_and_evidence(rng_key)
        )

        if self.mode == 'NLE':
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
            posterior_samples = nle_raw_samples
        else:
            assert self.theta_transform is not None
            flat_samples = np.asarray(
                self.theta_transform.adapter.to_array(self.posterior_samples)
            )
            weights = np.asarray(jnp.ones(flat_samples.shape[0]))
            posterior_samples = np.asarray(flat_samples)

        classic_raw_samples = self.classic_nested_samples.to_numpy()[:, :-3]
        classic_idx_weights = self.classic_nested_samples.index.to_numpy()
        classic_weights = np.asarray([el[1] for el in classic_idx_weights])

        with HidePrints():
            nle_samples = MCSamples(
                samples=posterior_samples,
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
            
            if self.mr_config.reference_theta is not None:
                markers = list(self.mr_config.reference_theta.values()) # type: ignore
            else:
                markers = None

            g = plots.get_subplot_plotter()
            g.triangle_plot(
                [nle_samples, classic_samples],
                filled=True,
                markers=markers, # type: ignore
                marker_args={'lw': 1}, # type: ignore
                legend_labels=[self.mode, 'Truth']
            )
        plt.savefig(
            self.mr_config.plot_save_dir
          + f'/corner_final.pdf',
            bbox_inches='tight'
        )

    def _add_to_simulation_pool(
            self,
            rng_key: NPKey,
            data: tuple[NDArray[np.float32], NDArray[np.bool_]],
            theta: dict[str, NDArray]
    ) -> None:
        start = self.current_round * self.mr_config.simulations_per_round
        end = (self.current_round + 1) * self.mr_config.simulations_per_round
        dmap, mask = data

        if self.mr_config.dequantise_data:
            _, dequantise_key = rng_key.split(2)
            dmap += dequantise_key.uniform(shape=dmap.shape, low=-0.5, high=0.5)

        self.ui.log(f'Adding data from idx {start} to idx {end}...')
        self.all_data[start:end, :] = dmap
        self.all_mask[start:end, :] = mask
        self.all_round_id[start:end] = self.current_round

        for key in self.all_theta.keys():
            self.all_theta[key][start:end] = theta[key]

    def _write_contiguous(self, storage, batch, start: int):
        def put(dst, src):
            # write src into dst at [start:start+src.shape[0]] along axis 0
            idx = (start,) + (0,) * (dst.ndim - 1)
            return lax.dynamic_update_slice(dst, src, idx)
        return jax.tree.map(put, storage, batch)

    def _sample_posterior(
            self, 
            posterior_key: PRNGKey, 
            n_samples: int
    ) -> dict[str, jnp.ndarray]:
        assert self.nflow is not None
        assert self.theta_transform is not None

        posterior_samples = self.nflow.sample_posterior(
            rng_key=posterior_key, 
            n_samples=n_samples,
            dmap_and_mask=(
                jax.device_put(self.reference_data),
                jax.device_put(self.reference_mask)
            ),
            check_proposal_probs=self.mr_config.check_proposal_probs,
            ui=self.ui
        )
        if posterior_samples is None:
            raise Exception('All posterior samples rejected.')
        corner(
            np.asarray(self.theta_transform.adapter.to_array(posterior_samples))
        )
        plt.savefig(
            self.mr_config.plot_save_dir
          + f'/corner_{self.current_round}.pdf',
            bbox_inches='tight'
        )
        return posterior_samples

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
            x0 = self.reference_data + epsilon
        else:
            x0 = self.reference_data

            # add batch dimension if none provided
            if x0.ndim == 1:
                x0 = x0[None, :]

        mask0 = np.asarray(self.reference_mask, dtype=bool)
        mask0 = np.broadcast_to(mask0, np.asarray(x0).shape)

        z0_np, zmask0_np, log_det_jac_np = self._transform_data_and_logdet(
            np.asarray(x0),
            mask0
        )
        z0 = jax.device_put(z0_np)
        zmask0 = jax.device_put(zmask0_np)
        log_det_jac = jax.device_put(log_det_jac_np)

        # log_det_jac will be of length n_requantisations, and since we
        # only add a scalar to each dim, the log_det_jac is the same for
        # each dequantisation
        if self.mr_config.dequantise_data:
            log_det_jac = log_det_jac[0]

        def lnlike_jax(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
            if self.theta_transform is not None:
                theta, _ = self.theta_transform(params, in_ns=True)
            else:
                raise NotImplementedError

            if not self.mr_config.dequantise_data:
                log_like = self.nflow.evaluate_lnlike( # type: ignore
                    theta[None, :], 
                    z0, 
                    mask=zmask0
                )
            else:
                log_like_by_permbatch = jax.vmap(
                    lambda zi: self.nflow.evaluate_lnlike( # type: ignore
                        theta[None, :],
                        zi[None, :],
                        mask=jnp.atleast_2d(zmask0)
                    )
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
          + f'/jax_samples_{self.current_round}.pdf',
            bbox_inches='tight'
        )

        self.lnZ_per_round.append(self.nested_samples.logZ())
        self.lnZerr_per_round.append(self.nested_samples.logZ(100).std()) # type: ignore

    def _make_train_val_set(
            self, 
            split_key: NPKey
    ) -> tuple[healpix_map_dataset_idx, healpix_map_dataset]:
        (trn_data, val_data), (trn_theta, val_theta), (trn_mask, val_mask), rnd_idx = self._split_train_val(
            split_key=split_key
        )

        assert not jnp.any(jnp.isnan(trn_data[trn_mask])) # no nans in unmasked data
        assert not any(jnp.isnan(arr).any() for arr in trn_theta.values())
        assert not jnp.any(jnp.isnan(val_data[val_mask]))
        assert not any(jnp.isnan(arr).any() for arr in val_theta.values())
        assert not jnp.any(rnd_idx == self.round_idx_badval)

        trn_set = healpix_map_dataset_idx(trn_data, trn_theta, trn_mask, rnd_idx)
        val_set = healpix_map_dataset(val_data, val_theta, val_mask)

        return trn_set, val_set

    def _split_train_val(
            self, 
            split_key: NPKey
    ) -> tuple[
        tuple[NDArray, NDArray], 
        tuple[dict[str, NDArray], dict[str, NDArray]],
        tuple[NDArray[np.bool_], NDArray[np.bool_]],
        NDArray
    ]:
        cur_max_idx = (self.current_round+1) * self.mr_config.simulations_per_round

        cur_data = self.all_data[:cur_max_idx, :]
        cur_data = cast(NDArray[np.float32], cur_data)

        cur_mask = self.all_mask[:cur_max_idx, :]
        cur_mask = cast(NDArray[np.bool_], cur_mask)

        cur_theta = {key: self.all_theta[key][:cur_max_idx] for key in self.all_theta.keys()}
        cur_round_idx = self.all_round_id[:cur_max_idx]

        return split_train_val_dict(
            key=split_key,
            y=cur_data,
            x=cur_theta,
            round_idx=cur_round_idx,
            mask=cur_mask,
            validation_fraction=self.train_config.validation_fraction
        )

    def _transform_data_and_logdet(
            self,
            data: NDArray,
            mask: NDArray
    ) -> tuple[NDArray, NDArray, NDArray]:
        '''
        Normalise only using stats computed from the training data.
        '''
        assert self.data_transform is not None
        (transformed_data, transformed_mask), log_det_jac = self.data_transform(data, mask)
        return transformed_data, transformed_mask, log_det_jac

    def _untransform_data_and_logdet(
            self,
            z: np.ndarray,
            mask: np.ndarray
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        assert self.data_transform is not None
        (data, mask), log_det_jac = self.data_transform.inverse_and_log_det(z, mask)
        return (data, mask), log_det_jac

    def _clear_data_summary_stats(self) -> None:
        if (self.data_transform is not None) and (self.theta_transform is not None):
            self.data_transform.clear()
            self.theta_transform.clear()

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

                if self.mode == 'NLE':
                    posterior_samples = self.nested_samples.sample(
                        n=n_posterior,
                        random_state=self.sample_posterior_seed
                    )
                    posterior_samples = self._reformat_samples(posterior_samples)

                elif self.mode == 'NPE':
                    assert self.posterior_samples is not None
                    tree = self.posterior_samples
                    n_available = next(iter(tree.values())).shape[0]
                    idx = np.random.choice(n_available, size=n_posterior, replace=True)
                    posterior_samples = {
                        key: value[idx] for key, value in tree.items()
                    }
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
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        return self.simulator_function(key, theta, True)

    def _instantiate_nflow(self) -> NeuralFlow:
        if (not self.train_config.restore_from_previous) or (self.nflow is None):
            nflow = NeuralFlow(
                self.target_ndim,
                config=self.nflow_config,
                data_transform=self.data_transform,
                theta_transform=self.theta_transform,
                embedding_net_config=self.embedding_net_config
            )
        else:
            nflow = self.nflow
        return nflow

    def _lnZ_plot(self) -> None:
        true_lnZ = self.true_lnZ
        true_lnZ_err = self.true_lnZerr
        lnZ_estimates = self.lnZ_per_round
        lnZ_estimates_err = self.lnZerr_per_round

        epochs = range(len(lnZ_estimates))
        _, ax = plt.subplots()

        ax.axhspan(
            true_lnZ - true_lnZ_err, # type: ignore
            true_lnZ + true_lnZ_err, # type: ignore
            color='gray', alpha=0.3, label='True lnZ ± err'
        )

        ax.errorbar(
            epochs, lnZ_estimates, yerr=lnZ_estimates_err,
            fmt='o-', capsize=4, label='Estimated lnZ'
        )

        ax.set_xlabel('Epoch')
        ax.set_ylabel('lnZ')
        ax.legend()
        plt.tight_layout()

        save_path = os.path.join(
            self.mr_config.plot_save_dir,
            'lnZ_evolution.pdf'
        )
        plt.savefig(save_path, bbox_inches='tight')

        # save epoch vs lnz
        array_save_path = os.path.join(
            self.mr_config.plot_save_dir,
            'epoch_lnZ.npy'
        )
        np.save(array_save_path, [lnZ_estimates, lnZ_estimates_err])
