# from surjectors package example

import argparse

from dataclasses import dataclass, fields
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
from dipolesbi.tools.maps import SimpleDipoleMap

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

from dipolesbi.tools.neural_flows import MAFNeuralLikelihood
from dipolesbi.tools.priors import DipolePrior
from dipolesbi.tools.simulator import Simulator


def make_model(dim, n_layers: int = 3, model="coupling"):
    def _bijector_fn(params):
        means, log_scales = unstack(params, -1)
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    def _flow(method, **kwargs):
        layers = []
        order = jnp.arange(dim)
        print(f'Using {n_layers} layers...')
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
    NSIDE = 2
    ndim=12 * NSIDE**2
    MEAN_DENSITY=120_000.
    OBSERVER_SPEED = 2.
    DIPOLE_LONGITUDE = 215.
    DIPOLE_LATITUDE = 40.
    theta0 = jnp.repeat(
        jnp.asarray(
            [[MEAN_DENSITY, OBSERVER_SPEED, DIPOLE_LONGITUDE, DIPOLE_LATITUDE]]
        ),
        repeats=n,
        axis=0
    )

    dipole = SimpleDipoleMap(NSIDE)    
    mean_count_range = [0.95*MEAN_DENSITY, 1.05*MEAN_DENSITY]
    prior = DipolePrior(mean_count_range=mean_count_range)
    prior.change_kwarg('N', 'mean_density')
    simulator = Simulator(prior, dipole.generate_dipole)
    prior_samples, y = simulator.make_batch_simulations(
        n_simulations=n,
        n_workers=12,
        simulation_batch_size=200
    )
    print(prior_samples)
    # prior_samples = jnp.asarray(np.random.uniform(low=1, high=50, size=n))
    # y = jnp.asarray(
    #     np.random.poisson(lam=prior_samples, size=(npix, n)),
    #     dtype=jnp.float32
    # ).T
    # prior_samples = jnp.expand_dims(prior_samples, -1) # (n, 1)
    prior_samples = jnp.asarray(prior_samples)
    y = jnp.asarray(y)
    y_mean = y.mean(); y_std = y.std()
    t_mean = prior_samples.mean(axis=0); t_std = prior_samples.std(axis=0)
    normalise_y = lambda input: (input - y_mean) / y_std
    normalise_t = lambda input: (input - t_mean) / t_std
    unnormalise_y = lambda input: y_std * input + y_mean
    unnormalise_t = lambda input: t_std * input + t_mean

    data = named_dataset(normalise_y(y), normalise_t(prior_samples))

    model = make_model(ndim, model=model, n_layers=5)
    params, losses = train(hk.PRNGSequence(2), data, model, n_iter)

    # theta1 = jnp.full(n, -2.)
    # theta2 = jnp.full(n, -4.)
    # theta = jnp.stack([theta1, theta2], axis=1)
    # theta = jnp.full(n, nbar)[:, None] # (n, 1)
    samples = model.apply(
        params,
        random.PRNGKey(2),
        method="sample",
        x=normalise_t(theta0)
    )

    plt.plot(losses)
    plt.show()

    # corner(np.asarray(samples), truths=ndim*[MEAN_DENSITY]) # labels=['$D_1$', '$D_2$'])
    # plt.show()
    # plt.hist(samples[:, 0])
    # plt.hist(samples[:, 1])
    # plt.show()

    # return theta0, y, model, unnormalise_y(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=1_000)
    parser.add_argument("--model", type=str, default="coupling")
    args = parser.parse_args()
    # thetas, y, model, samples = run(args.n_iter, args.model)
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
    NSIDE = 4
    ndim=12 * NSIDE**2
    MEAN_DENSITY=12_000.
    OBSERVER_SPEED = 2.
    DIPOLE_LONGITUDE = 215.
    DIPOLE_LATITUDE = 40.
    theta0 = jnp.repeat(
        jnp.asarray(
            [[MEAN_DENSITY, OBSERVER_SPEED, DIPOLE_LONGITUDE, DIPOLE_LATITUDE]]
        ),
        repeats=n,
        axis=0
    )

    dipole = SimpleDipoleMap(NSIDE)    
    mean_count_range = [0.95*MEAN_DENSITY, 1.05*MEAN_DENSITY]
    prior = DipolePrior(mean_count_range=mean_count_range)
    prior.change_kwarg('N', 'mean_density')
    simulator = Simulator(prior, dipole.generate_dipole)
    prior_samples, y = simulator.make_batch_simulations(
        n_simulations=n,
        n_workers=32,
        simulation_batch_size=200
    )
    # prior_samples = jnp.asarray(np.random.uniform(low=1, high=50, size=n))
    # y = jnp.asarray(
    #     np.random.poisson(lam=prior_samples, size=(npix, n)),
    #     dtype=jnp.float32
    # ).T
    # prior_samples = jnp.expand_dims(prior_samples, -1) # (n, 1)
    prior_samples = jnp.asarray(prior_samples)
    y = jnp.asarray(y)
    y_mean = y.mean(axis=0); y_std = y.std(axis=0) + 1e-8

    # t_mean = prior_samples.mean(axis=0); t_std = prior_samples.std(axis=0)

    normalise_y = lambda input: (input - y_mean) / y_std
    unnormalise_y = lambda input: y_std * input + y_mean

    @dataclass
    class NormData:
        theta_mean: jnp.ndarray
        theta_std: jnp.ndarray
        y_mean: jnp.ndarray
        y_std: jnp.ndarray
        data_ndim: int
        theta_ndim: int

        def __post_init__(self):
            for f in fields(self):
                if f.init:
                    assert f.shape == self.ndim


    mean_nbar = prior_samples[:, 0].mean()
    std_nbar = prior_samples[:, 0].std()
    mean_v = prior_samples[:, 1].mean()
    std_v = prior_samples[:, 1].std()

    t_mean = jnp.asarray([mean_nbar, mean_v, 0, 0])
    t_std = jnp.asarray([std_nbar, std_v, 1, 1])

    def transform_t(t):
        lon = t[..., 2]
        lat = t[..., 3]
        xyz = hp.ang2vec(lon, lat, lonlat=True)
        
        t_norm = (t - t_mean) / t_std
        t_transformed = jnp.stack(
            [
                t_norm[:, 0],
                t_norm[:, 1],
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2]
            ],
            axis=1
        )
        return t_transformed

    def untransform_t(t):
        x, y, z = t[:, 2], t[:, 3], t[:, 4]
        lon, lat = hp.vec2ang([x, y, z], lonlat=True)

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


    # normalise_t = lambda input: (input - t_mean) / t_std
    # unnormalise_t = lambda input: t_std * input + t_mean

    data = named_dataset(normalise_y(y), transform_t(prior_samples))

    nle = MAFNeuralLikelihood(ndim)
    losses = nle.train(hk.PRNGSequence(2), data)
    # model = make_model(ndim, model=args.model, n_layers=5)
    # params, losses = train(hk.PRNGSequence(2), data, model, args.n_iter)

    # theta1 = jnp.full(n, -2.)
    # theta2 = jnp.full(n, -4.)
    # theta = jnp.stack([theta1, theta2], axis=1)
    # theta = jnp.full(n, nbar)[:, None] # (n, 1)
    samples = nle.apply(random.PRNGKey(2), method="sample", x0=transform_t(theta0))
    # samples = nle.model.apply(
    #     params,
    #     random.PRNGKey(2),
    #     method="sample",
    #     x=transform_t(theta0)
    # )

    plt.plot(losses)
    plt.show()
    mean_samples = unnormalise_y(samples).mean(axis=0)

