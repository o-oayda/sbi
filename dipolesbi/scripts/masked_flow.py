# from surjectors package example

import argparse

import distrax
import haiku as hk
import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt
from corner import corner
import healpy as hp

from surjectors import (
    Chain,
    MaskedAutoregressive,
    MaskedCoupling,
    Permutation,
    TransformedDistribution,
)
from surjectors.nn import MADE, make_mlp
from surjectors.util import (
    as_batch_iterator,
    make_alternating_binary_mask,
    named_dataset,
    unstack,
)


def make_model(dim, n_layers: int = 3, model="coupling"):
    def _bijector_fn(params):
        means, log_scales = unstack(params, -1)
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    def _flow(method, **kwargs):
        layers = []
        order = jnp.arange(dim)
        for i in range(n_layers):
            if model == "coupling":
                print('Running masked coupling...')
                mask = make_alternating_binary_mask(dim, i % 2 == 0)
                layer = MaskedCoupling(
                    mask=mask,
                    bijector_fn=_bijector_fn,
                    conditioner=hk.Sequential(
                        [
                            make_mlp([8, 8, dim * 2]),
                            hk.Reshape((dim, 2)),
                        ]
                    ),
                )
                layers.append(layer)
            else:
                print('Running MAF...')
                layer = MaskedAutoregressive(
                    bijector_fn=_bijector_fn,
                    conditioner=MADE(
                        input_size=dim,
                        hidden_layer_sizes=[32,32],
                        n_params=2,
                        w_init=hk.initializers.TruncatedNormal(0.01),
                        b_init=jnp.zeros,
                    ),
                )
                order = order[::-1]
                layers.append(layer)
                layers.append(Permutation(order, 1))
        if model != "coupling":
            layers = layers[:-1]
        chain = Chain(layers)

        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(dim), jnp.ones(dim)),
            reinterpreted_batch_ndims=1,
        )
        td = TransformedDistribution(base_distribution, chain)
        return td(method=method, **kwargs)

    td = hk.transform(_flow)
    return td


def train(rng_seq, data, model, max_n_iter=1000):
    train_iter = as_batch_iterator(next(rng_seq), data, 100, True)
    params = model.init(next(rng_seq), method="log_prob", **train_iter(0))

    optimizer = optax.adam(1e-4)
    state = optimizer.init(params)

    @jax.jit
    def step(params, state, **batch):
        def loss_fn(params):
            lp = model.apply(params, None, method="log_prob", **batch)
            return -jnp.sum(lp)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_state = optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_state

    losses = np.zeros(max_n_iter)
    for i in range(max_n_iter):
        train_loss = 0.0
        for j in range(train_iter.num_batches):
            batch = train_iter(j)
            batch_loss, params, state = step(params, state, **batch)
            train_loss += batch_loss
        losses[i] = train_loss

    return params, losses


def run(n_iter, model):
    n = 10_000

    # 2D model with 2D data -> d_1 = 2*theta_1 + noise; d_2 = 2*theta_2 + noise
    # noise ~ N([0, 0], sigma=[10, 10]), diagonal cov.
    # ndim=2
    # thetas = distrax.Normal(jnp.zeros(2), jnp.full(2, 10)).sample(
    #     seed=random.PRNGKey(0), sample_shape=(n,)
    # )
    # y = 2 * thetas + distrax.Normal(jnp.zeros_like(thetas), 0.1).sample(
    #     seed=random.PRNGKey(1)
    # )
    # data = named_dataset(y, thetas)

    # healpix Poisson data
    npix=12
    ndim=npix
    nbar=10.

    prior_samples = jnp.asarray(np.random.uniform(low=1, high=50, size=n))
    y = jnp.asarray(
        np.random.poisson(lam=prior_samples, size=(npix, n)),
        dtype=jnp.float32
    ).T
    prior_samples = jnp.expand_dims(prior_samples, -1) # (n, 1)
    data = named_dataset(y, prior_samples)

    model = make_model(ndim, model=model)
    params, losses = train(hk.PRNGSequence(2), data, model, n_iter)

    # theta1 = jnp.full(n, -2.)
    # theta2 = jnp.full(n, -4.)
    # theta = jnp.stack([theta1, theta2], axis=1)
    theta = jnp.full(n, nbar)[:, None] # (n, 1)
    samples = model.apply(
        params,
        random.PRNGKey(2),
        method="sample",
        x=theta
    )

    plt.plot(losses)
    plt.show()

    corner(np.asarray(samples), truths=12*[nbar]) # labels=['$D_1$', '$D_2$'])
    plt.show()
    # plt.hist(samples[:, 0])
    # plt.hist(samples[:, 1])
    # plt.show()

    return theta, y, model, samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=1_000)
    parser.add_argument("--model", type=str, default="coupling")
    args = parser.parse_args()
    thetas, y, model, samples = run(args.n_iter, args.model)

