from jax.random import PRNGKey
from dipolesbi.tools.multiround_inferer import MultiRoundInferer
from dipolesbi.tools.configs import Scenario, DataTransformSpec, EmbeddingNetConfig
from dipolesbi.tools.inference import NotShitLikelihoodBasedInferer
from dipolesbi.tools.np_rngkey import npkey_from_jax
from dipolesbi.tools.maps import SimpleDipoleMap
import healpy as hp
import numpy as np
from dipolesbi.tools.priors_np import DipolePriorNP
import matplotlib.pyplot as plt
import os
import glob
import argparse
import jax


def lnZ_plot(inferer: MultiRoundInferer, out_dir: str) -> None:
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

    output_dir = out_dir
    folders = [
        f for f in glob.glob(os.path.join(output_dir, '*'))
        if os.path.isdir(f)
    ]
    if not folders:
        raise RuntimeError(f"No folders found in {output_dir}")
    latest_folder = max(folders, key=os.path.getctime)

    # Save the plot to the latest folder
    save_path = os.path.join(latest_folder, 'lnZ_evolution.png')
    plt.savefig(save_path, bbox_inches='tight')
    # plt.show()

    # save epoch vs lnz
    array_save_path = os.path.join(latest_folder, 'epoch_lnZ.npy')
    np.save(array_save_path, [lnZ_estimates, lnZ_estimates_err])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nside',
        type=int,
        help='Nside of simulated maps.'
    )
    parser.add_argument(
        '--mode',
        type=str,
        help='NLE or NPE.'
    )
    parser.add_argument(
        '--ssnle_seed',
        type=int,
        help='Integer seed for the multiround inferer pipeline.'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        help='Diretory for simulation outputs.'
    )
    args = parser.parse_args()

    x0_rng_key = PRNGKey(42)

    NSIDE = args.nside 
    TOTAL_SOURCES = 1_920_000
    MEAN_DENSITY = np.asarray(TOTAL_SOURCES / hp.nside2npix(NSIDE))
    MODE = args.mode
    theta0 = {
        'mean_density': MEAN_DENSITY,
        'observer_speed': np.asarray(2.),
        'dipole_longitude': np.asarray(215.),
        'dipole_latitude': np.asarray(40.)
    }

    model = SimpleDipoleMap(nside=NSIDE)
    # model.equatorial_plane_mask(angle=0)
    x0, mask = model.generate_dipole(npkey_from_jax(x0_rng_key), theta=theta0)
    model.reference_data = x0
    hp.projview(x0.squeeze() * mask.squeeze(), nest=True)
    plt.show()

    prior = DipolePriorNP(
        mean_count_range=[float(0.95*MEAN_DENSITY), float(1.05*MEAN_DENSITY)],
    )
    prior.change_kwarg('N', 'mean_density')
    prior_jax = prior.to_jax()
    adapter = prior_jax.get_adapter()

    # prior.change_kwarg('N', 'mean_density')
    # lb_inferer = NotShitLikelihoodBasedInferer(model.log_likelihood, prior, x0)
    # lb_inferer.run_ultranest()
    # corner(lb_inferer._samples)
    # plt.show()

    # embedding_cfg = EmbeddingNetConfig(
    #     nside=NSIDE,
    #     out_channels_per_layer=[2, 4, 8, 16],
    #     dropout_rate=0.2,
    # )
    # data_spec = DataTransformSpec.zscore(
    #     method='global', # use global for npe
    #     embedding_config=embedding_cfg,
    # )
    nside16_scenario_npe = Scenario.anynside_npe(
        nside=NSIDE,
        reference_theta=theta0,
        theta_adapter=adapter,
        theta_spec_overrides={'embed_transform_in_flow': True},
        multiround_overrides={
            'prng_integer_seed': args.ssnle_seed,
            'plot_save_dir': args.out_dir,
            'n_rounds': 1,
            'check_proposal_probs': True
        },
        training_overrides={'learning_rate': 0.001}
    )
    nside16_scenario_nle = Scenario.anynside_nle(
        nside=NSIDE,
        reference_theta=theta0,
        theta_adapter=adapter,
        multiround_overrides={
            'prng_integer_seed': args.ssnle_seed,
            'plot_save_dir': args.out_dir,
            'n_rounds': 1,
            'check_proposal_probs': True
        },
    )

    meta_cfg = {
        'NLE': nside16_scenario_nle,
        'NPE': nside16_scenario_npe
    }
    cur_cfg = meta_cfg[MODE]

    inferer = MultiRoundInferer(
        MODE, prior, model.generate_dipole, (x0, mask),
        multi_round_config=cur_cfg.multiround,
        transform_config=cur_cfg.transforms,
        nflow_config=cur_cfg.flow,
        train_config=cur_cfg.training
    )
    inferer.run()

    # lnZ_plot(inferer, out_dir=args.out_dir)
