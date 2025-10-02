from dataclasses import asdict
from functools import partial
from os import name
from typing import Callable, Literal, Optional
import distrax
import haiku as hk
from haiku._src.base import PRNGSequence
from haiku._src.transform import Transformed
from haiku._src.typing import PRNGKey
from jax import numpy as jnp
from jax._src.flatten_util import ravel_pytree
from numpy.typing import NDArray
from surjectors import (
    Chain,
    MaskedAutoregressive,
    Permutation,
    RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel,
    TransformedDistribution,
)
from surjectors.nn import MADE, make_mlp, make_transformer
from surjectors.util import unstack, named_dataset
from surjectors import AffineMaskedAutoregressiveInferenceFunnel
import optax
import jax
import numpy as np
import sys
import matplotlib.pyplot as plt
import logging
from abc import ABC, abstractmethod
from dipolesbi.tools.configs import EmbeddingNetConfig, NeuralFlowConfig, TrainingConfig, TransformConfig
from dipolesbi.tools.distributions import IndependentWrapper, NegBinomDist, PoissonDist, StudentT
from dipolesbi.tools.dataloader import as_batch_iterator_cpu2gpu, healpix_map_dataset, healpix_map_dataset_idx
from dipolesbi.tools.embedding_nets import HpCNNEmbedding
from dipolesbi.tools.np_rngkey import npkey_sequence_from_hk
from dipolesbi.tools.priors_jax import JaxPrior
from dipolesbi.tools.transforms import DipoleThetaTransform, InvertibleDataTransform, InvertibleThetaTransformJax
from dipolesbi.tools.hadamard_transform import HadamardTransform
from dipolesbi.tools.ui import MultiRoundInfererUI, NullMultiRoundInfererUI
from dipolesbi.tools.utils import convert_x_in_named_dataset, PytreeAdapter


def make_mlp_with_dropout(sizes, activation=jax.nn.silu, dropout_rate=0.0):
    def net(x, *, training: bool):
        for h in sizes[:-1]:
            x = hk.Linear(h)(x)
            x = activation(x)
            if dropout_rate and dropout_rate > 0.0:
                x = hk.dropout(hk.next_rng_key(), dropout_rate, x) if training else x
        return hk.Linear(sizes[-1])(x)
    return net  # callable: net(x, training=bool)

class AbstractNeuralFlow(ABC):
    class _ThetaTransformWrapper:
        """
        Because I mixed up the nomenclature in normalising flows, 
        I need a wrapper to reverse the transform functions.
        In normalising flows: forward means latent data -> x, inverse is x -> latent data.
        But this is the other way around in the transform,
        so this adapter makes theta transforms integrate cleanly into Surjectors chains.

        TODO: Later on, we'll rafactor to use the correct normalising flows names.
        """
        def __init__(self, base_transform: InvertibleThetaTransformJax):
            self._base = base_transform

        def forward_and_log_det(self, z, **kwargs):
            return self._base.inverse_and_log_det(z, **kwargs)

        def inverse_and_log_det(self, x, **kwargs):
            return self._base.forward_and_log_det(x, **kwargs)

    def __init__(self, config: NeuralFlowConfig) -> None:
        super().__init__()
        self.losses = []
        self.val_losses = []
        self.model = self.get_flow()
        self.best_params = None
        self._config = config
        self._target_is_pretransformed = not self._config.embed_target_transform_in_flow
        self._prior_adapter: Optional[PytreeAdapter] = None

    @property
    def nflow_config(self) -> NeuralFlowConfig:
        return self._config

    @abstractmethod
    def _flow(self, method: str, **kwargs) -> Transformed:
        pass

    @property
    @abstractmethod
    def mode(self) -> Literal['NLE'] | Literal['NPE']:
        pass

    @property
    @abstractmethod
    def data_transform(self) -> Optional[InvertibleDataTransform]:
        pass

    @property
    @abstractmethod
    def theta_transform(self) -> Optional[InvertibleThetaTransformJax]:
        pass

    def _clear_and_replace_stats(self, training_data: healpix_map_dataset_idx) -> None:
        # recompute summary stats for new training data
        if self.data_transform is not None:
            self.data_transform.clear()
            self.data_transform.compute_mean_and_std(training_data.y, training_data.mask)

        if self.theta_transform is not None:
            self.theta_transform.clear()
            self.theta_transform.compute_mean_and_std(training_data.x)

    def _maybe_transform_conditioning_variable(
            self,
            training_data: healpix_map_dataset_idx, 
            validation_data: healpix_map_dataset
    ) -> tuple[healpix_map_dataset_idx, healpix_map_dataset]:
        all_data = [training_data, validation_data]
        new_data = []

        for data in all_data:
            if (self.mode == 'NLE') and (self.theta_transform is not None):
                transformed_theta, _ = self.theta_transform(data.x)
                data = data._replace(x=transformed_theta)

            elif (self.mode == 'NPE') and (self.data_transform is not None):
                (transformed_data, transformed_mask), _ = self.data_transform(
                    data.y, data.mask
                )
                data = data._replace(y=transformed_data)
                data = data._replace(mask=transformed_mask)

            new_data.append(data)

        return tuple(new_data)

    @property
    def target_is_pretransformed(self) -> bool:
        return self._target_is_pretransformed

    def _maybe_pretransform_target(
            self,
            training_data: healpix_map_dataset_idx,
            validation_data: healpix_map_dataset
    ) -> tuple[healpix_map_dataset_idx, healpix_map_dataset]:
        all_data = [training_data, validation_data]
        new_data = []

        # pretransform if not part of the flow layers
        if self._target_is_pretransformed:
            for data in all_data:
                if (self.mode == 'NLE') and (self.data_transform is not None):
                    (transformed_data, transformed_mask), _ = self.data_transform(
                        data.y,
                        data.mask
                    )
                    data = data._replace(y=transformed_data)
                    data = data._replace(mask=transformed_mask)

                elif (self.mode == 'NPE') and (self.theta_transform is not None):
                    transformed_theta, _ = self.theta_transform(data.x)
                    data = data._replace(x=transformed_theta)

                new_data.append(data)
            return tuple(new_data)
        else:
            return tuple(all_data)

    def get_flow(self) -> Transformed:
        return hk.transform(self._flow)

    def _make_lr_schedule(
            self,
            warmup_steps: int,
            total_steps: int,
            base_lr: float,
            min_lr_ratio: float = 0.1
    ):
        assert warmup_steps < total_steps
        warmup = optax.linear_schedule(
            init_value=0., 
            end_value=base_lr,
            transition_steps=warmup_steps
        )
        cosine = optax.cosine_decay_schedule(
            init_value=base_lr,
            decay_steps=total_steps - warmup_steps,
            alpha=min_lr_ratio  # final LR = alpha * base_lr
        )
        return optax.join_schedules([warmup, cosine], [warmup_steps])

    def _make_optimiser(
            self,
            lr_schedule,
            clip_norm: float = 1.,
            weight_decay: float = 0.
    ):
        return optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
        )

    def plot_loss_curve(self, show: bool = True, save_path: Optional[str] = None) -> None:
        _, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(self.losses, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:orange'
        ax2.set_ylabel('Val Loss', color=color)
        ax2.plot(self.val_losses, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')

        if show:
            plt.show()

    def _make_loss_fn(
            self, 
            mode: Literal['NPE'] | Literal['NLE'],
            round_weights: Optional[jnp.ndarray] = None,
    ):
        def _nle_loss_fn(params, **batch):
            nlp = self._get_nlp_for_nle(params, **batch)
            if round_weights is not None:
                w = round_weights[batch['round_id']]
                w = w * (w.size / (jnp.sum(w) + 1e-12))
                return jnp.mean(w * nlp)
            else:
                return jnp.mean(nlp)

        def _npe_loss_fn(params, rng_key: PRNGKey, **batch):
            # in the named_dataset, y is data and x is theta (model params)
            nlp = self._get_nlp_for_npe(
                params,
                rng_key, 
                theta=batch['x'], 
                y=batch['y'], 
                is_training=True,
                mask=batch['mask']
            )
            return jnp.mean(nlp)

        if mode == 'NPE':
            return _npe_loss_fn
        elif mode == 'NLE':
            return _nle_loss_fn
        else:
            raise Exception(f'Mode {mode} not recognised.')

    def _get_nlp_for_nle(self, params, **batch):
        lp = self.model.apply(
            params, 
            None, 
            method='log_prob', 
            y=batch['y'],
            x=batch['x'],
            mask=batch['mask'],
            is_training=True
            # **self._model_kwargs(**batch)
        )
        return -lp

    # props to sbijax _src/npe.py
    def _get_nlp_for_npe(
            self, 
            params, 
            rng_key: PRNGKey, 
            theta, 
            y, 
            mask: jnp.ndarray,
            n_atoms=10, 
            is_training=True, 
    ):
        n = theta.shape[0]
        assert self.prior is not None

        n_atoms = np.maximum(2, np.minimum(n_atoms, n))
        repeated_y = jnp.repeat(y, n_atoms, axis=0)
        repeated_mask = jnp.repeat(mask, n_atoms, axis=0)
        probs = jnp.ones((n, n)) * (1 - jnp.eye(n)) / (n - 1)

        choice = partial(
            jax.random.choice, a=jnp.arange(n), replace=False, shape=(n_atoms - 1,)
        )
        sample_keys = jax.random.split(rng_key, probs.shape[0])
        choices = jax.vmap(lambda key, prob: choice(key, p=prob))(
            sample_keys, probs
        )
        contrasting_theta = theta[choices]

        atomic_theta = jnp.concatenate(
            (theta[:, None, :], contrasting_theta), axis=1
        )
        atomic_theta = atomic_theta.reshape(n * n_atoms, -1)

        log_prob_posterior = self.model.apply(
            params, 
            rng_key, 
            method="log_prob", 
            y=atomic_theta, 
            x=repeated_y, 
            is_training=is_training,
            mask=repeated_mask
        )
        log_prob_posterior = log_prob_posterior.reshape(n, n_atoms)

        # get x back out if the flow only sees z
        if self.target_is_pretransformed and self.theta_transform is not None:
            atomic_theta_raw, _ = self.theta_transform.inverse_and_log_det(atomic_theta)
        else:
            atomic_theta_raw = atomic_theta

        adapter = (
            self._prior_adapter if self._prior_adapter is not None
            else self.prior.get_adapter()
        )
        theta_tree = adapter.to_pytree(atomic_theta_raw)
        log_prob_prior = jax.vmap(self.prior.log_prob)(theta_tree)
        log_prob_prior = log_prob_prior.reshape(n, n_atoms)

        unnormalized_log_prob = log_prob_posterior - log_prob_prior
        log_prob_proposal_posterior = unnormalized_log_prob[
            :, 0
        ] - jax.scipy.special.logsumexp(unnormalized_log_prob, axis=-1)

        return -log_prob_proposal_posterior

    def _make_round_weights(self, n_rounds: int, alpha: float = 1.):
        t = jnp.arange(n_rounds)
        w = jnp.exp(alpha * (t - (n_rounds - 1)))
        return w / jnp.mean(w)

    def _model_kwargs(self, flip_order: bool = False, **batch):
        allowed = {"y", "x"}
        if flip_order:
            return {k: v for k, v in reversed(batch.items()) if k in allowed}
        else:
            return {k: v for k, v in batch.items() if k in allowed}

    def _model_init_params(self, rng_seq: PRNGSequence, train_iter):
        # a dict with keys 'x' (theta) and 'y' (data)
        initial_iter = train_iter(0)

        if 'mask' in initial_iter:
            self._ensure_mask_metadata(initial_iter['mask'])

        if self.mode == 'NPE':
            params = self.model.init(
                next(rng_seq), 
                method='log_prob', 
                y=initial_iter['x'],
                x=initial_iter['y'],
                mask=initial_iter['mask'],
                is_training=True
            )
        else:
            params = self.model.init(
                next(rng_seq), 
                method='log_prob', 
                y=initial_iter['y'],
                x=initial_iter['x'],
                mask=initial_iter['mask'],
                is_training=True
            )

        return params

    # TODO: this whole thing needs to be refactored (definitely not a part of neural flows)
    # we should probably be passing a mask object around with relevant stats
    def _ensure_mask_metadata(self, mask: jnp.ndarray | np.ndarray) -> None:
        if self._mask_metadata is not None:
            return

        mask_arr = jax.device_get(mask)
        mask_np = np.asarray(mask_arr)
        if mask_np.ndim == 0:
            mask_np = np.asarray([mask_np])
        if mask_np.ndim == 2:
            mask_np = mask_np[0]
        mask_np = mask_np.astype(bool, copy=False)

        mask_rev = mask_np[::-1]
        n_seen = int(mask_np.sum())
        keep_indices = np.where(mask_np)[0]

        healpix_counts: list[tuple[int, int]] = []
        if self.blocks is not None:
            dim = n_seen
            mask_start_idx = 0
            for n_keep_full, n_drop_full in self.blocks:
                drop_mask = mask_rev[mask_start_idx:mask_start_idx + n_drop_full]
                nd = int(drop_mask.sum())
                nk = int(dim - nd)
                healpix_counts.append((nk, nd))
                dim = nk
                mask_start_idx += n_drop_full

        self._mask_metadata = {
            'mask': mask_np,
            'mask_rev': mask_rev,
            'n_seen': n_seen,
            'healpix_counts': tuple(healpix_counts),
            'keep_indices': keep_indices,
        }

    def _get_step_fn(self, loss_fn, optimiser):
        if self.mode == 'NLE':
            @jax.jit
            def step(params, state, **batch):
                loss, grads = jax.value_and_grad(loss_fn)(params, **batch)
                updates, new_state = optimiser.update(grads, state, params)
                new_params = optax.apply_updates(params, updates)
                return loss, new_params, new_state
        elif self.mode == 'NPE':
            @jax.jit
            def step(params, state, rng_key: PRNGKey, **batch):
                loss, grads = jax.value_and_grad(loss_fn)(params, rng_key, **batch)
                updates, new_state = optimiser.update(grads, state, params)
                new_params = optax.apply_updates(params, updates)
                return loss, new_params, new_state
        else:
            raise Exception(f'{self.mode} not recognised.')

        return step

    def _get_val_step_fn(self):
        if self.mode == 'NLE':
            @jax.jit
            def val_step(params, **batch):
                return jnp.mean(self._get_nlp_for_nle(params, **batch))

        elif self.mode == 'NPE':
            @jax.jit
            def val_step(params, rng_key, **batch):
                return jnp.mean(
                    self._get_nlp_for_npe(
                        params, 
                        rng_key, 
                        theta=batch['x'], 
                        y=batch['y'],
                        mask=batch['mask'],
                        is_training=False
                    )
                )

        else:
            raise Exception(f'{self.mode} not recognised.')
            
        return val_step

    def train(
            self,
            rng_seq: hk.PRNGSequence, 
            training_data: healpix_map_dataset_idx, 
            validation_data: healpix_map_dataset,
            config: TrainingConfig = TrainingConfig(),
            ui: Optional[MultiRoundInfererUI | NullMultiRoundInfererUI] = None,
            prior: Optional[JaxPrior] = None
    ) -> tuple[NDArray, NDArray]:
        assert self.model is not None
        self.trn_config = config
        self.prior = prior
        if prior is not None:
            self._prior_adapter = prior.get_adapter()
        np_sequence = npkey_sequence_from_hk(rng_seq)

        self._clear_and_replace_stats(training_data)
        training_data, validation_data = self._maybe_transform_conditioning_variable(
            training_data, validation_data
        )
        self.training_data, self.validation_data = self._maybe_pretransform_target(
            training_data, validation_data
        )

        if ui:
            print_func = ui.log
        else:
            print_func = print

        theta_adapter = None
        if self.mode == 'NPE' and self.theta_transform is not None:
            theta_adapter = self.theta_transform.adapter

        # ensure theta is unravelled according to standard defined in prior
        train_iter = as_batch_iterator_cpu2gpu(
            rng_key=next(np_sequence), 
            data=convert_x_in_named_dataset(
                self.training_data,
                adapter=theta_adapter
            ),
            batch_size=self.trn_config.batch_size,
            shuffle=self.trn_config.shuffle_train
        )
        val_iter = as_batch_iterator_cpu2gpu(
            rng_key=next(np_sequence), 
            data=convert_x_in_named_dataset(
                self.validation_data,
                adapter=theta_adapter
            ),
            batch_size=self.trn_config.batch_size,
            shuffle=self.trn_config.shuffle_val
        )
        n_batches = train_iter.num_batches
        # assert n_batches == val_iter.num_batches
        steps_per_epoch = n_batches

        warmup_steps = max(
            10, int(self.trn_config.warmup_epochs * steps_per_epoch)
        )
        total_steps = max(
            steps_per_epoch * self.trn_config.max_n_iter, warmup_steps + 1
        )
        
        cur_sim_round = max(training_data.round_id) + 1
        if self.trn_config.weight_by_round:
            round_weights = self._make_round_weights(
                cur_sim_round, 
                self.trn_config.alpha_weight
            )
        else:
            round_weights = None
        print_func(f'Round weights: {round_weights}')

        if (self.trn_config.restore_from_previous) and (self.best_params is not None):
            print_func('Initialising using previously-inferred params...')
            params = self.best_params
        else:
            print_func('No previous state to initialise from. Starting fresh...')
            params = self._model_init_params(rng_seq, train_iter)

        lr_schedule = self._make_lr_schedule(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            base_lr=self.trn_config.learning_rate,
            min_lr_ratio=self.trn_config.min_lr_ratio
        )
        optimiser = self._make_optimiser(
            lr_schedule,
            clip_norm=self.trn_config.clip_norm,
            weight_decay=self.trn_config.weight_decay
        )
        # optimiser = optax.adam(self.trn_config.learning_rate, b2=self.trn_config.adam_b2)
        state = optimiser.init(params)
        loss_fn = self._make_loss_fn(self.mode, round_weights)
        step = self._get_step_fn(loss_fn, optimiser)
        val_step = self._get_val_step_fn()
        
        losses = np.nan * np.zeros(self.trn_config.max_n_iter)
        val_losses = np.nan * np.zeros(self.trn_config.max_n_iter)
        best_params = params
        best_val = np.inf
        wait = 0
        step_count = 0
        current_lr = (
            float(lr_schedule(step_count)) if lr_schedule is not None
            else self.trn_config.learning_rate
        )

        for i in range(self.trn_config.max_n_iter):
            train_loss = 0.0
            for j in range(train_iter.num_batches):
                batch = train_iter(j)

                # TODO: this is shit, refactor later
                if self.mode == 'NLE':
                    current_lr = float(lr_schedule(step_count))
                    batch_loss, params, state = step(params, state, **batch)
                else:
                    rng_key = next(rng_seq)
                    current_lr = float(lr_schedule(step_count))
                    batch_loss, params, state = step(params, state, rng_key, **batch)

                train_loss += float(batch_loss)
                step_count += 1
            train_loss /= max(1, train_iter.num_batches)
            
            val_loss = 0.0
            for j in range(val_iter.num_batches):
                batch = val_iter(j)

                if self.mode == 'NLE':
                    val_loss += float(val_step(params, **batch))
                else:
                    rng_key = next(rng_seq)
                    val_loss += float(val_step(params, rng_key, **batch))

            val_loss /= max(1, val_iter.num_batches)

            if ui:
                ui.set_subtitle(
                    f"Iter: {i} | "
                    f"Train NLL: {train_loss:.4f} | "
                    f"Val NLL: {val_loss:.4f} | "
                    f"𝜂 = {current_lr:.2e} | "
                    f"Stop at {self.trn_config.patience} ({wait})"
                )
            else:
                sys.stdout.write(
                    f"\rIter: {i} | "
                    f"Train NLL: {train_loss:.4f} | "
                    f"Val NLL: {val_loss:.4f} | "
                    f"𝜂 = {current_lr:.2e} | "
                    f"Stop at {self.trn_config.patience} ({wait})"
                )
                sys.stdout.flush()

            losses[i] = train_loss
            val_losses[i] = val_loss

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_params = params
                wait = 0
            else:
                wait += 1
                if i >= self.trn_config.min_n_iter and wait >= self.trn_config.patience:
                    print_func(
                        f"Early stopping at iteration {i} "
                        f"(best val NLL={best_val:.4f})"
                    )
                    break

        self.best_params = best_params
        self.losses = losses
        self.val_losses = val_losses

        return losses, val_losses

    def sample_likelihood_func(
            self, 
            rng_key: PRNGKey, 
            theta0: jnp.ndarray,
            mask: jnp.ndarray,
            **kwargs
    ) -> jnp.ndarray:
        samples = self.model.apply(
            self.best_params,
            rng_key,
            method='sample',
            x=theta0,
            is_training=False,
            mask=mask,
            **kwargs
        )
        if self.mode == 'NLE' and self.data_transform is not None:
            meta = self.get_mask_metadata(mask)
            keep_idxs_np = np.asarray(meta['keep_indices'], dtype=np.int32)
            keep_idxs = jnp.asarray(keep_idxs_np, dtype=jnp.int32)
            full_dim = mask.shape[-1] if mask.ndim > 1 else mask.shape[0]
            nan_value = jnp.array(jnp.nan, dtype=samples.dtype)
            full = jnp.full((samples.shape[0], full_dim), nan_value, dtype=samples.dtype)
            full = full.at[:, keep_idxs].set(samples)
            return full
        return samples

    def get_mask_metadata(self, mask: jnp.ndarray | np.ndarray | None = None) -> dict[str, object]:
        if self._mask_metadata is None:
            if mask is None:
                raise ValueError('Mask metadata requested before initialisation.')
            self._ensure_mask_metadata(mask)
        assert self._mask_metadata is not None
        return self._mask_metadata

    # props to sbijax _src/npe.py
    def sample_posterior(
            self,
            rng_key: PRNGKey,
            n_samples: int,
            dmap_and_mask: tuple[jnp.ndarray, jnp.ndarray],
            check_proposal_probs: bool = True,
            ui: Optional[MultiRoundInfererUI | NullMultiRoundInfererUI] = None,
            **kwargs
    ) -> dict[str, jnp.ndarray] | None:
        assert self.prior is not None
        assert self.mode == 'NPE'

        x0, data_mask = dmap_and_mask
        observable = jnp.atleast_2d(x0)
        data_mask = jnp.atleast_2d(data_mask)

        # this should be the same pointer to the nflow data_transform stats
        if self.mode == 'NPE' and self.data_transform is not None:
            # Bring to host for numpy-based transform, then back to jax
            (obs_np, obs_mask_np), _ = self.data_transform(
                np.asarray(jax.device_get(observable)),
                np.asarray(jax.device_get(data_mask))
            )
            observable = jnp.asarray(obs_np)
            data_mask = jnp.asarray(obs_mask_np)

        collected = []
        remaining_samples = int(n_samples)
        n_total_simulations_round = 0
        adapter = (
            self._prior_adapter if self._prior_adapter is not None
            else self.prior.get_adapter()
        )
        batch_size = 200

        if ui is not None:
            ui.begin_progress(total=n_samples)

        while remaining_samples > 0:
            current_batch_size = min(batch_size, remaining_samples)
            n_total_simulations_round += current_batch_size
            sample_key, rng_key = jax.random.split(rng_key)
            observable_tile = jnp.tile(observable, [batch_size, 1]) # (B, npix)
            observable_mask = jnp.tile(data_mask, [batch_size, 1])

            proposal = self.model.apply(
                self.best_params,
                sample_key,
                method="sample",
                # sample_shape=(batch_size,),
                x=observable_tile,
                mask=observable_mask,
                is_training=False,
                **kwargs
            )
            proposal = proposal[:current_batch_size]
            if self.target_is_pretransformed and self.theta_transform is not None:
                proposal, _ = self.theta_transform.inverse_and_log_det(proposal)
            proposal_tree = adapter.to_pytree(proposal)

            if check_proposal_probs:
                proposal_probs = self.prior.log_prob(proposal_tree)
                mask = jnp.isfinite(proposal_probs)
                proposal_tree = jax.tree_map(lambda a: a[mask], proposal_tree)
                n_accepted = int(mask.sum())
                if (ui is not None) and (current_batch_size - n_accepted == batch_size):
                    ui.log('All proposed samples discarded... breaking.')
                    return None
            else:
                n_accepted = current_batch_size

            if n_accepted > 0:
                collected.append(proposal_tree)
                remaining_samples -= n_accepted

            if ui is not None:
                ui.update_progress(n_accepted)

        concatenated_post_samples = jax.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0), *collected
        )
        concatenated_post_samples = jax.tree_map(
            lambda a: a[:n_samples], concatenated_post_samples
        )
        if ui is not None:
            ui.end_progress()
        return concatenated_post_samples

    def evaluate_lnlike(
        self,
        theta0: jnp.ndarray,
        x0: jnp.ndarray,
        mask: jnp.ndarray
    ) -> jnp.ndarray:
        logprob = self.model.apply(
            params=self.best_params,
            rng=None,  # deterministic evaluation
            method='log_prob',
            x=theta0,
            y=x0,
            mask=mask,
            is_training=False
        )
        return logprob

class MAFNeuralLikelihood(AbstractNeuralFlow):
    def __init__(
            self, 
            data_ndim: int, 
            n_layers: int = 3,
            conditioner_neurons_per_layer: int = 64,
            conditioner_n_hidden_layers: int = 2
    ) -> None:
        self.n_layers = n_layers
        self.data_ndim = data_ndim
        self.conditioner_neurons_per_layer = conditioner_neurons_per_layer
        self.conditioner_n_hidden_layers = conditioner_n_hidden_layers
        self.model = self.get_flow()

    def _bijector_fn(self, params):
        means, log_scales = unstack(params, -1)
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    def _flow(self, method: str, **kwargs):
        layers = []
        order = jnp.arange(self.data_ndim)
        n_neurons = self.conditioner_neurons_per_layer
        n_layers = self.conditioner_n_hidden_layers

        for _ in range(self.n_layers):
            layer = MaskedAutoregressive(
                bijector_fn=self._bijector_fn,
                conditioner=MADE(
                    input_size=self.data_ndim,
                    hidden_layer_sizes=n_layers * [n_neurons],
                    n_params=2, # mean and scale
                    w_init=hk.initializers.TruncatedNormal(0.01),
                    b_init=jnp.zeros,
                )
            )
            order = order[::-1]
            layers.append(layer)
            layers.append(Permutation(order, 1))

        layers = layers[:-1]
        chain = Chain(layers)

        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(self.data_ndim), jnp.ones(self.data_ndim)),
            reinterpreted_batch_ndims=1
        )
        td = TransformedDistribution(base_distribution, chain)
        return td(method, **kwargs)


class NeuralFlow(AbstractNeuralFlow):
    def __init__(
            self,
            target_ndim: int, 
            config: NeuralFlowConfig,
            data_transform: Optional[InvertibleDataTransform] = None,
            theta_transform: Optional[InvertibleThetaTransformJax] = None,
            embedding_net_config: Optional[EmbeddingNetConfig] = None
    ) -> None:
        '''
        If using a heirarchical flow, pass an invertible data transform so
        the detail blocks can be accessed.
        '''
        super().__init__(config)
        self.target_ndim = target_ndim
        self._data_transform = data_transform
        self._theta_transform = theta_transform
        self._mode: Literal['NLE', 'NPE'] = self.nflow_config.mode # NLE or NPE
        self.embedding_net_config = embedding_net_config
        
        if isinstance(self.data_transform, HadamardTransform):
            self.blocks = self.data_transform.blocks
        else:
            self.blocks = None

        self._mask_metadata: dict[str, object] | None = None
        self.model = self.get_flow()

    @property
    def mode(self) -> Literal['NLE', 'NPE']:
        return self._mode

    @property
    def data_transform(self) -> Optional[InvertibleDataTransform]:
        return self._data_transform

    @property
    def theta_transform(self) -> Optional[InvertibleThetaTransformJax]:
        return self._theta_transform

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for i, layer in enumerate(getattr(self, "layers", [])):
            # Name of the layer
            name = layer.__class__.__name__
            spec = ""
            if hasattr(layer, "n_keep"):
                spec += f"n_keep={layer.n_keep} "
            if hasattr(layer, "n_drop"):
                spec += f"n_drop: {layer.n_drop}"
            lines.append(f"  [{i:02d}] {name} {spec}")
        lines.append(")")
        return "\n".join(lines)

    def _bijector_fn(self, params):
        means, log_scales = unstack(params, -1) # happens on non-surjective layer in heirarchical
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    def _base_distribution_fn(self, n_dim):
        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(n_dim), jnp.ones(n_dim)),
            reinterpreted_batch_ndims=1,
        )
        return base_distribution

    def _conditioner_fn(self, input_dim, output_dim, **kwargs):
        if self.nflow_config.conditioner == "mlp":
            return make_mlp(
                [self.nflow_config.conditioner_n_neurons]
              * self.nflow_config.conditioner_n_layers
              + [output_dim],
            )
        elif self.nflow_config.conditioner == "transformer":
            return make_transformer(
                {
                    'output_dim': output_dim,
                    'num_heads': 4,
                    'num_layers': 4,
                    'key_size': 32,
                    'dropout_rate': 0.1,
                    'widening_factor': 4,
                    **kwargs
                }
            )
        elif self.nflow_config.conditioner == 'made':
            return MADE(
                input_size=input_dim,
                hidden_layer_sizes=[self.nflow_config.conditioner_n_neurons]
                    * self.nflow_config.conditioner_n_layers,
                n_params=2,
                w_init=hk.initializers.TruncatedNormal(stddev=0.01),
                b_init=jnp.zeros,
                activation=jax.nn.tanh
            )

        logging.fatal("didnt find correct conditioner type")
        raise ValueError("didnt find correct conditioner type")

    def _decoder_fn(
            self, 
            n_dimension: int, 
            decoder_distribution: str,
            eps: float = 1e-8,
            integer_transform: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    ):
        decoder_params_lookup = {
            'gaussian': 2,
            'nb': 2,
            'poisson': 1,
            'students_t': 3
        }
        self.n_decoder_params = decoder_params_lookup[decoder_distribution]
        assert self.nflow_config.decoder_n_neurons is not None
        assert self.nflow_config.decoder_n_layers is not None

        decoder_net = make_mlp(
            [self.nflow_config.decoder_n_neurons] * self.nflow_config.decoder_n_layers
          + [n_dimension * self.n_decoder_params],
            activation=jax.nn.tanh,
        )

        def _fn(z):
            # we get a distribution per pixel, e.g. 500 latent pixels => 500 mus for Gaussian
            params = decoder_net(z)
            if decoder_distribution == 'gaussian':
                mu, log_scale = jnp.split(params, 2, -1)
                return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)), 1)

            elif decoder_distribution == 'nb':
                mu_raw, r_raw = jnp.split(params, 2, axis=-1)
                mu = jax.nn.softplus(mu_raw) + eps
                r  = jax.nn.softplus(r_raw) + eps
                dist = NegBinomDist(mu, r)
                return IndependentWrapper(dist, reinterpreted_batch_ndims=1)

            elif decoder_distribution == 'poisson':
                log_lambda = params
                log_lambda = jnp.clip(log_lambda, -10, 10)
                lam = jnp.exp(log_lambda)
                dist = PoissonDist(lam)
                return IndependentWrapper(
                    dist, 
                    reinterpreted_batch_ndims=1, 
                    integer_transform=integer_transform
                )

            elif decoder_distribution == 'students_t':
                mu, log_scale, log_df = jnp.split(params, 3, -1)
                scale = jnp.exp(log_scale)
                df = jnp.exp(log_df) + 2.0     # keep df > 2 for finite variance
                dist = StudentT(df=df, loc=mu, scale=scale) # type: ignore
                return IndependentWrapper(dist, reinterpreted_batch_ndims=1)

            else:
                raise Exception('Unrecognised decoder distribution.')

        return _fn

    def _flow(self, method: str, **kwargs):
        return self._make_flow(method, **kwargs)

    def _get_surjective_layer(
            self, 
            name: str
    ) -> (
            AffineMaskedAutoregressiveInferenceFunnel
          | RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel
        ):
        name_to_class = {
            'affine_MAF': AffineMaskedAutoregressiveInferenceFunnel,
            'rational_quadratic_MAF': (
                RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel
            )
        }
        return name_to_class[name]
    
    def _healpix_funnel(
            self, 
            surjective_layer_type: (
                AffineMaskedAutoregressiveInferenceFunnel
              | RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel
            ), 
            layers: list, 
            blocks: list[tuple[int, int]],
            mask_metadata: Optional[dict[str, object]],
            in_dim: int,
            integer_transform: Optional[
                Callable[
                    [int | Literal['all']],
                    Callable[[jnp.ndarray], jnp.ndarray]
                ]
            ],
            one_and_done: bool = False,
            maf_extension: Optional[int] = None
    ) -> tuple[list, int]:
        if mask_metadata is None:
            raise ValueError('Mask metadata must be initialised before building the healpix funnel.')

        mask_r = np.asarray(mask_metadata['mask_rev'])
        n_seen = int(mask_metadata['n_seen'])
        dim = n_seen

        if one_and_done:
            assert maf_extension is None
            n_kp = blocks[-1][0]
            n_drp = dim - n_kp
            assert n_kp + n_drp == dim
            blocks = [(n_kp, n_drp)]

        if maf_extension:
            assert maf_extension % 2 == 0
        else:
            maf_extension = 0

        mask_start_idx = 0
        for lvl, (n_keep_full, n_drop_full) in enumerate(blocks):
            # n_keep and n_dropped split data into y_minus and y_plus, as in
            # y_plus, y_minus = y[..., : self.n_keep], y[..., self.n_keep :]
            # however, we need to be privy to the mask on the data; the data
            # will be truncated, and we don't want to drop off more pixels
            # than are valid at that level
            nd = int(mask_r[mask_start_idx:mask_start_idx+n_drop_full].sum())
            nk = int(dim - nd)

            surjective_layer = surjective_layer_type( # type: ignore
                n_keep=nk,
                decoder=self._decoder_fn(
                    n_dimension=nd,
                    decoder_distribution=self.nflow_config.decoder_distribution,
                    integer_transform=(
                        integer_transform(lvl) if integer_transform else None
                    )
                ),
                conditioner=self._conditioner_fn(
                    input_dim=nk,
                    output_dim=2 * nk
                )
            )
            setattr(surjective_layer, 'n_drop', nd) # for __repr__
            layers.append(surjective_layer)
            dim = nk
            mask_start_idx += n_drop_full

            # bijective extension
            for _ in range(maf_extension):
                layers.append(
                    MaskedAutoregressive(
                        bijector_fn=self._bijector_fn,
                        conditioner=self._conditioner_fn(
                            input_dim=dim,
                            output_dim=2 * dim
                        )
                    )
                )
                layers = self._reverse_perm(layers, dim)
                
        out_dim = dim
        return layers, int(out_dim)

    def _reverse_perm(self, layers: list, dim: int) -> list:
        order = jnp.arange(dim)
        order = order[::-1]
        layers.append(Permutation(order, 1))
        return layers

    def _maybe_add_transform_to_layers(self, layers: list) -> list:
        assert len(layers) == 0

        # if an NLE, we want to add the data (target) transform to the first layer
        # if an NPE, we want the theta (target) transform on the first layer
        if not self.target_is_pretransformed:
            if self.mode == 'NLE':
                if self.data_transform is not None:
                    assert not isinstance(self.data_transform, HadamardTransform), (
                        'Hadamard transform will not work as a flow layer. '
                        'Use as a pre-transform instead.'
                    )
                    layers.append(self.data_transform)

            elif self.mode == 'NPE':
                if self.theta_transform is not None:
                    layers.append(self._ThetaTransformWrapper(self.theta_transform))

            else:
                raise Exception(f'Mode {self.mode} not recognised.')

        return layers

    def _maybe_transform_healpy_map(self, **kwargs) -> tuple[dict, jnp.ndarray]:
        '''
        If the healpix data is the target (NLE), we want to mask out pixels
        such that these pixels are truncated i.e. never seen by the flow.
        Instead, if the data is the conditioning variable (NPE), we potentially
        want to pass the mask and the full data vector to the CNN embedding network.
        '''
        is_training = kwargs.pop("is_training", False)
        mask = kwargs.pop('mask', None)
        assert mask is not None, 'Mask should be hard-coded in dataset.'

        # x being the data here
        if self.mode == 'NPE' and 'x' in kwargs and self.embedding_net_config:
            self.embedding_network = HpCNNEmbedding(**asdict(self.embedding_net_config))
            kwargs = dict(kwargs)
            x_in = kwargs['x']

            if x_in.ndim == 3:
                # for some unknown reason, an extra dimension is added at the start
                # of x when sampling, e.g. (2, 200, 3072) = (?, B, npix)
                # we flatten over this axis to get around the problem and repeat
                # the mask too
                x_in = x_in.reshape((x_in.shape[0] * x_in.shape[1], x_in.shape[2]))
                jax.debug.print('{mask}', mask=mask)
                mask = jnp.repeat(mask, repeats=2, axis=0)
                jax.debug.print('{mask}', mask=mask)

            kwargs['x'] = self.embedding_network(x_in, mask, is_training=is_training)

        # y being the data here
        elif self.mode == 'NLE' and self.data_transform is not None and 'y' in kwargs:
            # assume all masks are the same across batches
            assert self._mask_metadata is not None, (
                'Mask metadata must be initialised before applying the data transform.'
            )
            keep_idxs_np = self._mask_metadata.get('keep_indices') if self._mask_metadata else None
            if keep_idxs_np is not None:
                keep_idxs = jnp.asarray(keep_idxs_np, dtype=jnp.int32)
            else:
                keep_idxs = self.data_transform.keep_idxs
            kwargs['y'] = jnp.take(kwargs['y'], keep_idxs, axis=1)

        return kwargs, mask

    def _make_flow(self, method: str, **kwargs):
        assert self.nflow_config.architecture is not None

        # if NLE, masked pixels are always truncated
        kwargs, batch_mask = self._maybe_transform_healpy_map(**kwargs)

        if 'y' in kwargs:
            cur_dim = kwargs['y'].shape[-1]
        elif self.mode == 'NLE' and self._mask_metadata is not None:
            cur_dim = int(self._mask_metadata['n_seen'])
        else:
            cur_dim = self.target_ndim
        self.layers = []
        self.layers = self._maybe_add_transform_to_layers(self.layers)

        for layer in self.nflow_config.architecture:
            if layer == 'MAF':
                self.layers.append(
                    MaskedAutoregressive(
                        bijector_fn=self._bijector_fn,
                        conditioner=self._conditioner_fn(
                            input_dim=cur_dim,
                            output_dim=2 * cur_dim
                        )
                    )
                )
                self.layers = self._reverse_perm(self.layers, cur_dim)

            elif layer == 'healpix_funnel':
                assert self.nflow_config.surjective_layer_type is not None
                assert self.nflow_config.funnel_one_and_done is not None
                assert self.nflow_config.funnel_maf_extension is not None
                assert self.blocks is not None

                surjective_layer_type = self._get_surjective_layer(
                    self.nflow_config.surjective_layer_type
                )
                # add surjective healpix funnel
                if self._mask_metadata is None:
                    self._ensure_mask_metadata(batch_mask)

                self.layers, cur_dim = self._healpix_funnel(
                    surjective_layer_type=surjective_layer_type, 
                    layers=self.layers, 
                    blocks=self.blocks, 
                    mask_metadata=self._mask_metadata,
                    in_dim=cur_dim, 
                    integer_transform=None,
                    one_and_done=self.nflow_config.funnel_one_and_done,
                    maf_extension=self.nflow_config.funnel_maf_extension
                )
                # don't add an extra perm --- added by the maf extension
                if self.nflow_config.funnel_maf_extension == 0:
                    self.layers = self._reverse_perm(self.layers, cur_dim)

            elif layer == 'surjective_MAF':
                assert self.nflow_config.data_reduction_factor is not None
                n_keep = int(self.nflow_config.data_reduction_factor * cur_dim)
                n_drop = cur_dim - n_keep
                assert n_keep + n_drop == cur_dim

                surjective_layer_type = self._get_surjective_layer(
                    self.nflow_config.surjective_layer_type
                )
                surjective_layer = surjective_layer_type( # type: ignore
                    n_keep=n_keep,
                    decoder=self._decoder_fn(
                        n_dimension=n_drop,
                        decoder_distribution=self.nflow_config.decoder_distribution,
                        integer_transform=None
                    ),
                    conditioner=self._conditioner_fn(
                        input_dim=n_keep,
                        output_dim=2 * n_keep
                    )
                )
                setattr(surjective_layer, 'n_drop', n_drop) # for __repr__
                self.layers.append(surjective_layer)
                cur_dim = n_keep
                self.layers = self._reverse_perm(self.layers, cur_dim)

            else:
                raise Exception(f'Layer {layer} not recognised.')

        # remove redundant perm at end of stack
        self.layers = self.layers[:-1]

        chain = Chain(self.layers)
        td = TransformedDistribution(self._base_distribution_fn(cur_dim), chain)
        return td(method, **kwargs)
