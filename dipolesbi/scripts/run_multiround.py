from jax.random import PRNGKey, split
from dipolesbi.scripts.evidence_comparison import MultiRoundInferer
from dipolesbi.tools.priors_jax import DipolePriorJax
from dipolesbi.tools.maps import SimpleDipoleMapJax
import healpy as hp
from jax import numpy as jnp


if __name__ == '__main__':
    rng_key = PRNGKey(42)

    NSIDE = 8
    TOTAL_SOURCES = 1_920_000
    MEAN_DENSITY = jnp.asarray(TOTAL_SOURCES / hp.nside2npix(NSIDE))
    theta0 = {
        'mean_density': MEAN_DENSITY,
        'observer_speed': jnp.asarray(2.),
        'dipole_longitude': jnp.asarray(215.),
        'dipole_latitude': jnp.asarray(40.)
    }

    model = SimpleDipoleMapJax(nside=NSIDE)
    x0_key, rng_key = split(rng_key)
    x0 = model.generate_dipole(x0_key, theta=theta0)

    prior = DipolePriorJax(
        mean_count_range=[float(0.95*MEAN_DENSITY), float(1.05*MEAN_DENSITY)]
    )
    inferer = MultiRoundInferer(
        rng_key, prior, model.generate_dipole, x0,
        train_config={'learning_rate': 1e-4}
    )
    inferer.run()
