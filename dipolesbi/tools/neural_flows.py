import distrax
import haiku as hk
from haiku._src.typing import PRNGKey
from jax import numpy as jnp
from numpy.typing import NDArray
from surjectors import (
    Chain,
    MaskedAutoregressive,
    Permutation,
    TransformedDistribution,
)
from surjectors.nn import MADE
from surjectors.util import as_batch_iterator, unstack, named_dataset
from optax._src.base import Params
import optax
import jax
import numpy as np
import sys


class MAFNeuralLikelihood:
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

    def get_flow(self):
        return hk.transform(self._flow)

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

    def train(
            self,
            rng_seq: hk.PRNGSequence, 
            data: named_dataset, 
            max_n_iter: int = 1000,
            batch_size: int = 100,
            learning_rate: float = 0.001
    ) -> NDArray:
        train_iter = as_batch_iterator(
            rng_key=next(rng_seq), 
            data=data,
            batch_size=batch_size,
            shuffle=True
        )
        params = self.model.init(next(rng_seq), method='log_prob', **train_iter(0))

        optimiser = optax.adam(learning_rate)
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

        losses = np.zeros(max_n_iter)
        for i in range(max_n_iter):
            train_loss = 0.0
            for j in range(train_iter.num_batches):
                batch = train_iter(j)
                batch_loss, params, state = step(params, state, **batch)
                train_loss += batch_loss
            sys.stdout.write(f"\rTraining loss (iteration {i}): {train_loss:.4f}")
            sys.stdout.flush()
            losses[i] = train_loss

        self.params = params
        self.losses = losses
        return losses

    def apply(self, rng_key: PRNGKey, method: str, x0):
        samples = self.model.apply(
            self.params,
            rng_key,
            method=method,
            x=x0
        )
        return samples
