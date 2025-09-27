import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jax.random import PRNGKey

from dipolesbi.scripts.bijectors import LatitudeBijector, UniformIntervalSigmoid


def test_uniform_interval_sigmoid_round_trip():
    bijector = UniformIntervalSigmoid(low=-1.0, high=3.0)

    z = jnp.array([-4.0, 0.0, 1.5])
    theta, logdet_f = bijector.forward_and_log_det(z)

    z_recon, logdet_inv = bijector.inverse_and_log_det(theta)

    expected_theta = bijector.low + bijector.span * sigmoid(z)
    assert jnp.allclose(theta, expected_theta, atol=1e-6)
    assert jnp.allclose(z_recon, z, atol=1e-6)
    assert jnp.allclose(logdet_f + logdet_inv, jnp.zeros_like(logdet_f), atol=1e-6)

def test_uniform_interval_sigmoid_round_trip_from_theta():
    bijector = UniformIntervalSigmoid(low=0., high=1000.)
    prng_key = PRNGKey(42)
    theta = jax.random.uniform(prng_key, minval=0., maxval=1000., shape=(10_000,))

    z, logdet_inv = bijector.inverse_and_log_det(theta)
    theta_recon, logdet_f = bijector.forward_and_log_det(z)

    assert jnp.allclose(theta_recon, theta, atol=1e-6)
    assert jnp.allclose(logdet_f + logdet_inv, jnp.zeros_like(logdet_f), atol=2e-4) # high for fp32

# probably to reimplement soon

# def test_uniform_interval_sigmoid_inverse_out_of_support_logdet():
#     bijector = UniformIntervalSigmoid(low=0.0, high=1.0)
#
#     theta = jnp.array([-0.25, 1.5])
#     _, logdet = bijector.inverse_and_log_det(theta)
#
#     assert jnp.all(jnp.isneginf(logdet))


def test_latitude_bijector_round_trip():
    bijector = LatitudeBijector()

    z = jnp.array([-3.0, -0.2, 0.5, 2.5])
    b, logdet_f = bijector.forward_and_log_det(z)

    z_recon, logdet_inv = bijector.inverse_and_log_det(b)

    assert jnp.allclose(z_recon, z, atol=1e-6)
    assert jnp.allclose(logdet_f + logdet_inv, jnp.zeros_like(logdet_f), atol=1e-6)


# def test_latitude_bijector_inverse_out_of_support_logdet():
#     bijector = LatitudeBijector()
#
#     b = jnp.array([-2.0, 2.0])
#     _, logdet = bijector.inverse_and_log_det(b)
#
#     assert jnp.all(jnp.isneginf(logdet))
