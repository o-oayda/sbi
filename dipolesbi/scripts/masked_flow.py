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
import healpy as hp
from dipolesbi.tools.dataloader import split_train_val
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=1_000)
    parser.add_argument("--model", type=str, default="coupling")
    args = parser.parse_args()
    n = 20_000

    # healpix Poisson data
    NSIDE = 8
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
    prior_samples = jnp.asarray(prior_samples)
    y = jnp.asarray(y)
    (y_tr, y_val), (t_tr, t_val) = split_train_val(y, prior_samples)
    
    y_mean = y_tr.mean(axis=0); y_std = y_tr.std(axis=0) + 1e-8
    normalise_y = lambda input: (input - y_mean) / y_std
    unnormalise_y = lambda input: y_std * input + y_mean
    
    mean_nbar = t_tr[:, 0].mean()
    std_nbar = t_tr[:, 0].std()
    mean_v = t_tr[:, 1].mean()
    std_v = t_tr[:, 1].std()

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

    train_data = named_dataset(normalise_y(y_tr), transform_t(t_tr))
    val_data = named_dataset(normalise_y(y_val), transform_t(t_val))

    nle = MAFNeuralLikelihood(ndim)
    nle.train(hk.PRNGSequence(2), train_data, val_data)
    nle.plot_loss_curve()

    samples = nle.apply(random.PRNGKey(2), method="sample", x0=transform_t(theta0))

    mean_samples = unnormalise_y(samples).mean(axis=0)
    hp.projview(mean_samples, nest=True, graticule=True)
    plt.show()

