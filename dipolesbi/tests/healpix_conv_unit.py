from dataclasses import asdict
from dipolesbi.tools.embedding_nets import HpCNNEmbedding
from dipolesbi.tools.maps import SimpleDipoleMapJax
from jax import numpy as jnp
from jax.random import PRNGKey
from dipolesbi.tools.configs import EmbeddingNetConfig
import haiku as hk
import jax
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np


def _example_dipole_map():
    reference_theta = {
        'mean_density': jnp.array(150.),
        'observer_speed': jnp.array(2.),
        'dipole_longitude': jnp.array(160.),
        'dipole_latitude': jnp.array(-20.)
    }
    rng_key = PRNGKey(0)
    nside = 32

    dipole = SimpleDipoleMapJax(nside=nside)
    dipole_map = dipole.generate_dipole(rng_key, reference_theta)
    return jnp.atleast_2d(dipole_map)

def _conv_config() -> EmbeddingNetConfig:
    return EmbeddingNetConfig(nside=32, output_dim=48)

def test_conv_shape():
    dipole_map = _example_dipole_map()
    cfg = asdict(_conv_config())

    def apply(x):
        conv = HpCNNEmbedding(**cfg)
        mask = jnp.ones(x.shape[:2], dtype=x.dtype)
        return conv(x, mask)

    tf = hk.transform(apply)
    rng = jax.random.PRNGKey(42)

    params = tf.init(rng, dipole_map)
    output = tf.apply(params, rng, dipole_map)

    output_np = np.asarray(output).flatten()
    hp.projview(output_np)
    plt.savefig('out_conv.pdf')

    assert output.shape == (1, cfg['output_dim'])
