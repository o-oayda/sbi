from jax.random import PRNGKey, split
from dipolesbi.scripts.evidence_comparison import MultiRoundInferer
from dipolesbi.tools.configs import SurjectiveNLEConfig, TrainingConfig
from dipolesbi.tools.priors_jax import DipolePriorJax
from dipolesbi.tools.maps import SimpleDipoleMapJax
import healpy as hp
from jax import numpy as jnp
from dipolesbi.tools.transforms import HaarWaveletTransform


if __name__ == '__main__':
    rng_key = PRNGKey(42)
    # nle_config = SurjectiveNLEConfig.standard(n_layers=8)

    NSIDE = 16
    TOTAL_SOURCES = 1_920_000
    MEAN_DENSITY = jnp.asarray(TOTAL_SOURCES / hp.nside2npix(NSIDE))
    theta0 = {
        'mean_density': MEAN_DENSITY,
        'observer_speed': jnp.asarray(2.),
        'dipole_longitude': jnp.asarray(215.),
        'dipole_latitude': jnp.asarray(40.)
    }

    haar_transform = HaarWaveletTransform(first_nside=NSIDE, last_nside=1)

    model = SimpleDipoleMapJax(nside=NSIDE)
    x0_key, rng_key = split(rng_key)
    x0 = model.generate_dipole(x0_key, theta=theta0)

    prior = DipolePriorJax(
        mean_count_range=[float(0.95*MEAN_DENSITY), float(1.05*MEAN_DENSITY)]
    )

    nle_config = SurjectiveNLEConfig.heirarchical(
        blocks=haar_transform._build_surjective_blocks(),
        maf_stack_size=15
    )
    # low learning rate high nside?
    train_config = TrainingConfig(patience=20, learning_rate=1e-4)

    inferer = MultiRoundInferer(
        rng_key, prior, model.generate_dipole, x0,
        reference_theta=theta0,
        simulation_budget=20_000,
        n_rounds=10,
        custom_data_transform=haar_transform,
        nle_config=nle_config,
        train_config=train_config
    )
    inferer.run()
