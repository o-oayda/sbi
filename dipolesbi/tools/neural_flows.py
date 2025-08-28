import distrax
import haiku as hk
from haiku._src.transform import Transformed
from haiku._src.typing import PRNGKey
from healpy import npix2nside
from jax import numpy as jnp
from numpy.typing import NDArray
from surjectors import (
    Chain,
    MaskedAutoregressive,
    Permutation,
    TransformedDistribution,
)
from surjectors.nn import MADE, make_mlp, make_transformer
from surjectors.util import as_batch_iterator, unstack, named_dataset
from surjectors import AffineMaskedAutoregressiveInferenceFunnel
import optax
import jax
import numpy as np
import sys
import matplotlib.pyplot as plt
import logging
from abc import ABC, abstractmethod
from dipolesbi.tools.configs import SurjectiveNLEConfig, TrainingConfig
from dipolesbi.tools.distributions import IndependentWrapper, NegBinomDist, PoissonDist, StudentT
from dipolesbi.tools.healpix_helpers import build_layer_perms, first_layer_stratifying_perm, get_healpix_superpixels, make_latent_dims, permute_within_types

def make_mlp_with_dropout(sizes, activation=jax.nn.silu, dropout_rate=0.0):
    def net(x, *, training: bool):
        for h in sizes[:-1]:
            x = hk.Linear(h)(x)
            x = activation(x)
            if dropout_rate and dropout_rate > 0.0:
                x = hk.dropout(hk.next_rng_key(), dropout_rate, x) if training else x
        return hk.Linear(sizes[-1])(x)
    return net  # callable: net(x, training=bool)

class NeuralLikelihood(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.losses = []
        self.val_losses = []
        self.model = self.get_flow()

    @abstractmethod
    def _flow(self, method: str, **kwargs) -> Transformed:
        pass

    def get_flow(self) -> Transformed:
        return hk.transform(self._flow)

    def plot_loss_curve(self) -> None:
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

        plt.show()

    def train(
            self,
            rng_seq: hk.PRNGSequence, 
            training_data: named_dataset, 
            validation_data: named_dataset,
            config: TrainingConfig = TrainingConfig()
    ) -> tuple[NDArray, NDArray]:
        assert self.model is not None
        self.trn_config = config

        train_iter = as_batch_iterator(
            rng_key=next(rng_seq), 
            data=training_data,
            batch_size=self.trn_config.batch_size,
            shuffle=True
        )
        val_iter = as_batch_iterator(
            rng_key=next(rng_seq), 
            data=validation_data,
            batch_size=self.trn_config.batch_size,
            shuffle=True
        )
        params = self.model.init(next(rng_seq), method='log_prob', **train_iter(0))

        optimiser = optax.adam(self.trn_config.learning_rate, b2=self.trn_config.adam_b2)
        state = optimiser.init(params)
        
        @jax.jit
        def step(params, state, **batch):
            def loss_fn(params):
                lp = self.model.apply(params, None, method="log_prob", **batch)
                return -jnp.sum(lp)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_state = optimiser.update(grads, state, params)
            new_params = optax.apply_updates(params, updates)
            return loss, new_params, new_state
        
        @jax.jit
        def val_step(params, **batch):
            return -jnp.mean(
                self.model.apply(params, None, method='log_prob', **batch)
            )

        losses = np.nan * np.zeros(self.trn_config.max_n_iter)
        val_losses = np.nan * np.zeros(self.trn_config.max_n_iter)
        best_params = params
        best_val = np.inf
        wait = 0

        for i in range(self.trn_config.max_n_iter):
            train_loss = 0.0
            for j in range(train_iter.num_batches):
                batch = train_iter(j)
                batch_loss, params, state = step(params, state, **batch)
                train_loss += batch_loss
            
            val_loss = 0.0
            for j in range(val_iter.num_batches):
                batch = val_iter(j)
                val_loss += val_step(params, **batch)
            val_loss /= max(1, val_iter.num_batches)

            sys.stdout.write(
                f"\rIteration: {i} | "
                f"Training NLL: {train_loss:.4f} | "
                f"Validation NLL: {val_loss:.4f} | "
                f"Early stopping at {self.trn_config.patience} ({wait})"
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
                if wait >= self.trn_config.patience:
                    print(
                        f"\nEarly stopping at iteration {i} "
                        f"(best val NLL={best_val:.4f})"
                    )
                    break

        self.params = best_params
        self.losses = losses
        self.val_losses = val_losses

        return losses, val_losses

    def sample_likelihood_func(
            self, 
            rng_key: PRNGKey, 
            theta0: jnp.ndarray
    ) -> jnp.ndarray:
        samples = self.model.apply(
            self.params,
            rng_key,
            method='sample',
            x=theta0
        )
        return samples

    def evaluate_lnlike(
        self,
        theta0: jnp.ndarray,
        x0: jnp.ndarray
    ) -> jnp.ndarray:
        logprob = self.model.apply(
            params=self.params,
            rng=None, # don't pass key
            method='log_prob',
            x=theta0,
            y=x0
        )
        return logprob

class MAFNeuralLikelihood(NeuralLikelihood):
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


class MAFSurjectiveNeuralLikelihood(NeuralLikelihood):
    def __init__(
            self,
            data_ndim: int, 
            config: SurjectiveNLEConfig = SurjectiveNLEConfig()
    ) -> None:
        self.data_ndim = data_ndim
        self.nle_config = config
        self.model = self.get_flow()

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for i, layer in enumerate(getattr(self, "layers", [])):
            # Name of the layer
            name = layer.__class__.__name__
            # Try to pull out a useful dimension or config
            if hasattr(layer, "n_keep"):
                spec = f"n_keep={layer.n_keep}"
            elif hasattr(layer, "perm"):
                spec = f"perm shape={layer.perm.shape}"
            else:
                spec = ""
            lines.append(f"  [{i:02d}] {name} {spec}")
        lines.append(")")
        return "\n".join(lines)

    def _bijector_fn(self, params):
        means, log_scales = unstack(params, -1)
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    def _base_distribution_fn(self, n_dim):
        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(n_dim), jnp.ones(n_dim)),
            reinterpreted_batch_ndims=1,
        )
        return base_distribution

    def _conditioner_fn(self, input_dim, output_dim, **kwargs):
        if self.nle_config.conditioner == "mlp":
            return make_mlp(
                [self.nle_config.conditioner_n_neurons]
              * self.nle_config.conditioner_n_layers
              + [output_dim],
            )
        elif self.nle_config.conditioner == "transformer":
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
        elif self.nle_config.conditioner == 'made':
            return MADE(
                input_size=input_dim,
                hidden_layer_sizes=[self.nle_config.conditioner_n_neurons]
                    * self.nle_config.conditioner_n_layers,
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
            eps: float = 1e-8
    ):
        decoder_params_lookup = {
            'gaussian': 2,
            'nb': 2,
            'poisson': 1,
            'students_t': 3
        }
        self.n_decoder_params = decoder_params_lookup[decoder_distribution]
        # decoder_net = make_mlp_with_dropout(
        #     [self.decoder_n_neurons] * self.decoder_n_layers
        #   + [n_dimension * self.n_decoder_params],
        #     dropout_rate=0.1
        # )
        decoder_net = make_mlp(
            [self.nle_config.decoder_n_neurons] * self.nle_config.decoder_n_layers
          + [n_dimension * self.n_decoder_params],
            activation=jax.nn.tanh,
        )

        def _fn(z):
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
                lambda_raw = params
                lam = jax.nn.softplus(lambda_raw) + eps
                dist = PoissonDist(lam)
                return IndependentWrapper(dist, reinterpreted_batch_ndims=1)

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
        if self.nle_config.flow_type == 'heirarchical':
            # print('Using heirarchical flow...')
            return self._heirarchical_flow(method, **kwargs)
        elif self.nle_config.flow_type == 'standard':
            # print('Using standard flow...')
            return self._standard_flow(method, **kwargs)
        elif self.nle_config.flow_type == 'coarse':
            # print('Using coarse flow...')
            return self._coarse_flow(method, **kwargs)
        else:
            raise Exception(
                f'Flow type {self.nle_config.flow_type} in config not recognised.'
            )

    
    def _standard_flow(self, method: str, **kwargs):
        assert self.nle_config.data_reduction_factor is not None
        assert self.nle_config.n_layers is not None

        dim = self.data_ndim
        nside = npix2nside(dim)
        super_pixel_blocks = get_healpix_superpixels(nside)
        latentdim = int(self.nle_config.data_reduction_factor * dim)

        perm0 = first_layer_stratifying_perm(latentdim, super_pixel_blocks)
        self.layers = []
        self.layers.append(Permutation(perm0, 1))

        latent_dims = make_latent_dims(
            dim,
            self.nle_config.n_layers, 
            self.nle_config.data_reduction_factor
        )
        layer_perms = build_layer_perms(latent_dims)

        for i in range(self.nle_config.n_layers):
            reduc = self.nle_config.data_reduction_factor
            latent_dim = int(reduc * dim)
            layer = AffineMaskedAutoregressiveInferenceFunnel(
                n_keep=latent_dim,
                decoder=self._decoder_fn(
                    dim - latent_dim,
                    decoder_distribution=self.nle_config.decoder_distribution
                ),
                conditioner=MADE(
                    input_size=latent_dim,
                    hidden_layer_sizes=[self.nle_config.conditioner_n_neurons]
                    * self.nle_config.conditioner_n_layers,
                    n_params=2,
                    w_init=hk.initializers.TruncatedNormal(stddev=0.01),
                    b_init=jnp.zeros,
                    activation=jax.nn.tanh,
                ),
            )
            self.layers.append(layer)
            dim = latent_dim
            self.layers.append(Permutation(layer_perms[i], 1))

        self.layers = self.layers[:-1]
        chain = Chain(self.layers)

        td = TransformedDistribution(self._base_distribution_fn(dim), chain)
        return td(method, **kwargs)

    def _heirarchical_flow(self, method: str, **kwargs):
        # coarse_block = self.blocks[0]
        # detail_blocks = self.blocks[1:]
        assert len(self.nle_config.blocks) > 0
        assert self.nle_config.maf_stack_size is not None

        self.layers = []
        dim0 = self.data_ndim

        dim = dim0
        self.drop_idxs = []
        self.keep_idxs = []
        dropped_total = 0       

        for _, (per, nk, nd) in enumerate(self.nle_config.blocks):
            n_drop = nd
            n_keep = nk
            perm = per

            self.layers.append(Permutation(perm, 1))

            self.layers.append(
                AffineMaskedAutoregressiveInferenceFunnel(
                    n_keep=n_keep,
                    decoder=self._decoder_fn(
                        n_dimension=n_drop,
                        decoder_distribution=self.nle_config.decoder_distribution
                    ),
                    conditioner=self._conditioner_fn(
                        input_dim=n_keep,
                        output_dim=2 * n_keep
                    )
                )
            )

            dim = n_keep
            dropped_total += n_drop

        # add a MAF bijective stack at the end
        for _ in range(self.nle_config.maf_stack_size):
            self.layers.append(
                MaskedAutoregressive(
                    bijector_fn=self._bijector_fn,
                    conditioner=self._conditioner_fn(
                        input_dim=dim,
                        output_dim=2 * dim
                    )
                )
            )
            order = jnp.arange(dim)
            order = order[::-1]
            self.layers.append(Permutation(order, 1))

        if self.nle_config.maf_stack_size > 0:
            self.layers = self.layers[:-1]

        chain = Chain(self.layers)
        td = TransformedDistribution(self._base_distribution_fn(dim), chain)
        return td(method, **kwargs)

    def _coarse_flow(self, method: str, **kwargs):
        assert self.nle_config.data_reduction_factor is not None
        assert self.nle_config.n_coarse is not None
        assert self.nle_config.n_layers is not None

        dim0 = self.data_ndim
        reduc = self.nle_config.data_reduction_factor

        cur_dim = dim0
        through_dims = []
        self.layers = []

        for i in range(self.nle_config.n_layers):
            if cur_dim > self.nle_config.n_coarse:
                keep = max(self.nle_config.n_coarse, int(reduc * cur_dim))
                through_dims.append(keep)
                drop = cur_dim - keep

                conditioner = self._conditioner_fn(
                    input_dim=keep,
                    output_dim=2 * keep
                )
                layer = AffineMaskedAutoregressiveInferenceFunnel(
                    n_keep=keep,
                    decoder=self._decoder_fn(
                        drop,
                        decoder_distribution=self.nle_config.decoder_distribution
                    ),
                    conditioner=conditioner
                )
                self.layers.append(layer)
                cur_dim = keep
                perm = permute_within_types(cur_dim, self.nle_config.n_coarse, seed=141+i)
                self.layers.append(Permutation(perm, 1))
            else:
                order = jnp.arange(cur_dim)
                through_dims.append(cur_dim)
                conditioner = self._conditioner_fn(
                    input_dim=cur_dim,
                    output_dim=2 * cur_dim
                )
                layer = MaskedAutoregressive(
                    bijector_fn=self._bijector_fn,
                    conditioner=conditioner
                )

                order = order[::-1]
                self.layers.append(layer)
                self.layers.append(Permutation(order, 1))
                self.layers.append(layer)

            self.layers = self.layers[:-1]

        print(f'Bijective data dimensions: {through_dims}')

        if self.nle_config.permute_data:
            self.layers = self.layers[:-1]
        chain = Chain(self.layers)

        td = TransformedDistribution(self._base_distribution_fn(cur_dim), chain)
        return td(method, **kwargs)
