from jax.random import PRNGKey
from dipolesbi.tools.multiround_inferer import MultiRoundInferer
from dipolesbi.tools.configs import ConfigOfConfigs
from dipolesbi.tools.inference import NotShitLikelihoodBasedInferer
from dipolesbi.tools.np_rngkey import npkey_from_jax
from dipolesbi.tools.maps import SimpleDipoleMap
import healpy as hp
import numpy as np
from dipolesbi.tools.priors_np import DipolePriorNP
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

    NSIDE = 32
    TOTAL_SOURCES = 1_920_000
    MEAN_DENSITY = np.asarray(TOTAL_SOURCES / hp.nside2npix(NSIDE))
    theta0 = {
        'mean_density': MEAN_DENSITY,
        'observer_speed': np.asarray(2.),
        'dipole_longitude': np.asarray(215.),
        'dipole_latitude': np.asarray(40.)
    }

    model = SimpleDipoleMap(nside=NSIDE)
    x0 = model.generate_dipole(npkey_from_jax(rng_key), theta=theta0)
    model.reference_data = x0

    prior = DipolePriorNP(
        mean_count_range=[float(0.95*MEAN_DENSITY), float(1.05*MEAN_DENSITY)],
    )

    # prior.change_kwarg('N', 'mean_density')
    # lb_inferer = NotShitLikelihoodBasedInferer(model.log_likelihood, prior, x0)
    # lb_inferer.run_ultranest()
    # corner(lb_inferer._samples)
    # plt.show()

    nside16_config = ConfigOfConfigs.nside16(theta0)
    nside32_config = ConfigOfConfigs.nside32(theta0)
    meta_cfg = {
        16: nside16_config,
        32: nside32_config
    }
    cur_cfg = meta_cfg[NSIDE]

    inferer = MultiRoundInferer(
        rng_key, prior, model.generate_dipole, x0,
        multi_round_config=cur_cfg.multiround_config,
        nle_config=cur_cfg.ssnle_config,
        train_config=cur_cfg.training_config
    )
    inferer.run()

    lnZ_plot(inferer)
