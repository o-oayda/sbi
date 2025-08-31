from jax.random import PRNGKey, split
from dipolesbi.scripts.evidence_comparison import MultiRoundInferer
from dipolesbi.tools.configs import MultiRoundInfererConfig, SurjectiveNLEConfig, TrainingConfig
from dipolesbi.tools.priors_jax import DipolePriorJax
from dipolesbi.tools.maps import SimpleDipoleMapJax
import healpy as hp
from jax import numpy as jnp
from dipolesbi.tools.transforms import HaarWaveletTransform, ZScore
import matplotlib.pyplot as plt
import os
import glob


def lnZ_plot(inferer: MultiRoundInferer) -> None:
    true_lnZ = inferer.true_lnZ
    true_lnZ_err = inferer.true_lnZerr
    lnZ_estimates = inferer.lnZ_per_round
    lnZ_estimates_err = inferer.lnZerr_per_round

    epochs = range(len(lnZ_estimates))
    _, ax = plt.subplots()

    ax.axhspan(
        true_lnZ - true_lnZ_err, # type: ignore
        true_lnZ + true_lnZ_err, # type: ignore
        color='gray', alpha=0.3, label='True lnZ ± err'
    )

    ax.errorbar(
        epochs, lnZ_estimates, yerr=lnZ_estimates_err,
        fmt='o-', capsize=4, label='Estimated lnZ'
    )

    ax.set_xlabel('Epoch')
    ax.set_ylabel('lnZ')
    ax.legend()
    plt.tight_layout()

    output_dir = 'nle_out'
    folders = [f for f in glob.glob(os.path.join(output_dir, '*')) if os.path.isdir(f)]
    if not folders:
        raise RuntimeError(f"No folders found in {output_dir}")
    latest_folder = max(folders, key=os.path.getctime)

    # Save the plot to the latest folder
    save_path = os.path.join(latest_folder, 'lnZ_evolution.png')
    plt.savefig(save_path, bbox_inches='tight')

    plt.show()

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

    # possibly widening and deepening the decoder network improves
    # convergence to true lnZ -> e.g. 3000 pixels dropped w low n_neurons not great
    # nle_config = SurjectiveNLEConfig.standard(
    #     n_layers=10,
    #     data_reduction_factor=0.75,
    #     conditioner_n_layers=4,
    #     conditioner_n_neurons=256,
    #     decoder_n_layers=4,
    #     decoder_n_neurons=512,
    # )
    nle_config = SurjectiveNLEConfig.heirarchical(
        blocks=haar_transform._build_surjective_blocks(n_chunks=1),
        maf_stack_size=15,
        conditioner_n_layers=4,
        conditioner_n_neurons=256,
        decoder_n_layers=4,
        decoder_n_neurons=512,
        decoder_distribution='gaussian'
    )
    # low learning rate high nside?
    train_config = TrainingConfig(
        patience=20, 
        learning_rate=1e-5, 
        restore_from_previous=True
    )
    mr_config = MultiRoundInfererConfig(
        simulation_budget=50_000,
        n_rounds=15,
        custom_data_transform=haar_transform,
        # custom_data_transform=ZScore(),
        reference_theta=theta0,
        dequantise_data=True,
        n_requantisations=32
    )

    inferer = MultiRoundInferer(
        rng_key, prior, model.generate_dipole, x0,
        multi_round_config=mr_config,
        nle_config=nle_config,
        train_config=train_config
    )
    inferer.run()

    lnZ_plot(inferer)
