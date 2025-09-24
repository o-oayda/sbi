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
    theta0 = {
        'mean_density': MEAN_DENSITY,
        'observer_speed': np.asarray(2.),
        'dipole_longitude': np.asarray(215.),
        'dipole_latitude': np.asarray(40.)
    }

    model = SimpleDipoleMap(nside=NSIDE)
    x0 = model.generate_dipole(npkey_from_jax(x0_rng_key), theta=theta0)
    model.reference_data = x0

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

    embedding_cfg = EmbeddingNetConfig(
        nside=NSIDE,
        out_channels_per_layer=[2, 4, 8, 16],
        dropout_rate=0.2,
    )
    data_spec = DataTransformSpec.zscore(
        method='batchwise',
        embedding_config=embedding_cfg,
    )
    # data_spec = DataTransformSpec.hadamard(
    #     first_nside=NSIDE,
    #     embed_in_flow=False
    # )
    nside16_scenario = Scenario.nside16_npe(
        reference_theta=theta0,
        theta_adapter=adapter,
        data_spec=data_spec,
        theta_spec_overrides={'embed_transform_in_flow': True},
        flow_overrides={
            'conditioner_n_neurons': 64,
            'conditioner_n_layers': 2,
            'architecture': 5 * ['MAF'],
        },
        training_overrides={'restore_from_previous': False},
        multiround_overrides={
            'prng_integer_seed': args.ssnle_seed,
            'dequantise_data': False,
            'n_requantisations': None,
            'plot_save_dir': args.out_dir,
            'initial_fraction': 0.,
            'simulation_budget': 50_000,
            'n_rounds': 3,
        },
    )
    # nside16_scenario = Scenario.nside16_nle(
    #     theta0, 
    #     theta_adapter=adapter,
    #     data_transform_overrides={
    #         'matrix_type': 'hadamard',
    #         'normalise_details': True,
    #         'last_nside': 1 # broken now?
    #     },
    #     training_overrides={'restore_from_previous': False},
    #     nflow_overrides={
    #         'decoder_distribution': 'gaussian',
    #         'architecture': ['healpix_funnel'], # + 2 * ['MAF'],# + 14 * ['MAF'],
    #         'funnel_one_and_done': False,
    #         'funnel_maf_extension': 2
    #     },
    #     multiround_overrides={
    #         'prng_integer_seed': args.ssnle_seed,
    #         'dequantise_data': False,
    #         'n_requantisations': None,
    #         'plot_save_dir': args.out_dir,
    #         'initial_fraction': 0.,
    #         'simulation_budget': 50_000
    #     }
    # )
    meta_cfg = {
        4: nside16_scenario,
        16: nside16_scenario,
    }
    cur_cfg = meta_cfg[NSIDE]

    inferer = MultiRoundInferer(
        'NPE', prior, model.generate_dipole, x0,
        multi_round_config=cur_cfg.multiround,
        transform_config=cur_cfg.transforms,
        nflow_config=cur_cfg.flow,
        train_config=cur_cfg.training
    )
    inferer.run()

    lnZ_plot(inferer, out_dir=args.out_dir)
