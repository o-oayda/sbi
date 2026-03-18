import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse

from catsim import RacsLow3, RacsLow3Config
from catsim.utils.healsphere import downgrade_ignore_nan
import healpy as hp
import numpy as np

from dipolesbi.tools.configs import DataTransformSpec, Scenario
from dipolesbi.tools.multiround_inferer import MultiRoundInferer
from dipolesbi.tools.np_rngkey import NPKey
from dipolesbi.tools.priors_np import DipolePriorNP
from dipolesbi.tools.ui import MultiRoundInfererUI
from dipolesbi.tools.utils import batch_simulate


def _parse_modes(raw_modes: list[str] | None, parser: argparse.ArgumentParser) -> list[str]:
    modes: list[str] = []
    for entry in raw_modes or []:
        modes.extend(part.strip().upper() for part in entry.split(",") if part.strip())
    if not modes:
        parser.error("At least one mode must be provided via --mode.")
    return modes


def _build_real_sample(
    model: RacsLow3,
    flux_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    catalogue = model.catalogue
    flux = np.asarray(catalogue["Total_flux"], dtype=np.float64)
    ra = np.asarray(catalogue["RA"], dtype=np.float64)
    dec = np.asarray(catalogue["Dec"], dtype=np.float64)

    valid = (
        np.isfinite(flux)
        & np.isfinite(ra)
        & np.isfinite(dec)
        & (flux >= flux_min)
    )
    pixel_indices = hp.ang2pix(
        model.nside,
        ra[valid],
        dec[valid],
        lonlat=True,
        nest=True,
    ).astype(np.int64, copy=False)

    density_map = np.bincount(
        pixel_indices,
        minlength=hp.nside2npix(model.nside),
    ).astype(np.float32, copy=False)
    mask = model.mask_map.astype(np.bool_, copy=False)
    density_map = density_map.copy()
    density_map[~mask] = np.nan

    if model.downscale_nside is None:
        return density_map, mask

    coarse_map, coarse_mask = downgrade_ignore_nan(
        density_map,
        mask,
        model.downscale_nside,
    )
    coarse_map = coarse_map.astype(np.float32, copy=False)
    coarse_mask = coarse_mask.astype(np.bool_, copy=False)
    coarse_map = coarse_map.copy()
    coarse_map[~coarse_mask] = np.nan
    return coarse_map, coarse_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        nargs="+",
        help='One or more modes to run, separated by spaces (e.g. "--mode NLE NPE").',
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        required=True,
        help="Number of simulations to run.",
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=15,
        help="Number of rounds of inference.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of local workers to use for simulation batches.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save outputs into.",
    )
    parser.add_argument(
        "--ssnle_seed",
        type=int,
        default=0,
        help="Seed used for sequential neural estimators.",
    )
    parser.add_argument(
        "--downscale_nside",
        type=int,
        default=None,
        help="Optional HEALPix nside to downscale maps to.",
    )
    parser.add_argument(
        "--no_ui",
        action="store_true",
        help="Disable the Rich multi-round progress UI.",
    )
    parser.add_argument(
        "--flux_min",
        type=float,
        default=2.0,
        help="RACS-low3 flux threshold in mJy.",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=64,
        help="Native HEALPix nside used by the simulator.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50_000,
        help="Chunk size used inside the simulator.",
    )
    parser.add_argument(
        "--catalogue_path",
        type=str,
        default=None,
        help="Optional path to the RACS-low3 catalogue.",
    )
    parser.add_argument(
        "--alpha_mean",
        type=float,
        default=0.8,
        help="Mean of the Gaussian spectral-index model.",
    )
    parser.add_argument(
        "--alpha_sigma",
        type=float,
        default=0.2,
        help="Standard deviation of the Gaussian spectral-index model.",
    )
    parser.add_argument(
        "--fractional_error_flux_min_mjy",
        type=float,
        default=10.0,
        help="Minimum flux used when building the empirical fractional-error lookup.",
    )
    args = parser.parse_args()

    modes = _parse_modes(args.mode, parser)

    config = RacsLow3Config(
        flux_min=args.flux_min,
        nside=args.nside,
        chunk_size=args.chunk_size,
        use_float32=False,
        downscale_nside=args.downscale_nside,
        store_final_samples=True,
        catalogue_path=args.catalogue_path,
        alpha_mean=args.alpha_mean,
        alpha_sigma=args.alpha_sigma,
        fractional_error_flux_min_mjy=args.fractional_error_flux_min_mjy,
    )
    model = RacsLow3(config)
    model.initialise_data()

    x0, mask = _build_real_sample(model, flux_min=args.flux_min)
    effective_nside = hp.npix2nside(x0.size)

    observed_count = float(np.nansum(x0))
    if observed_count <= 0:
        raise ValueError("Observed RACS-low3 map has zero total counts after masking/cuts.")

    prior = DipolePriorNP(
        mean_count_range=[
            np.log10(max(1.0, 0.5 * observed_count)),
            np.log10(2.0 * observed_count),
        ],
        speed_range=[0, 8],
    )
    prior.change_kwarg(
        param_short_name="N",
        new_kwarg="log10_n_initial_samples",
    )

    theta_0 = {
        "log10_n_initial_samples": np.log10(observed_count),
        "observer_speed": 1.0,
        "dipole_longitude": 264.021,
        "dipole_latitude": 48.253,
    }

    def simulator_wrapper(
        rng_key: NPKey | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        return model.generate_dipole(rng_key=rng_key, **kwargs)

    prior_jax = prior.to_jax()

    for mode in modes:
        if mode == "NPE":
            scenario = Scenario.anynside_npe(
                nside=effective_nside,
                theta_prior=prior_jax,
                reference_theta=theta_0,
                theta_spec_overrides={"embed_transform_in_flow": True},
                multiround_overrides={
                    "prng_integer_seed": args.ssnle_seed,
                    "plot_save_dir": args.out_dir,
                    "n_rounds": args.n_rounds,
                    "simulation_budget": args.n_simulations,
                    "likelihood_chunk_size_gb": 0.5,
                    "n_likelihood_samples": 10_000,
                },
                training_overrides={"learning_rate": 0.001},
            )
        elif mode == "NLE":
            data_spec = DataTransformSpec.zscore(method="batchwise")
            scenario = Scenario.anynside_nle(
                nside=effective_nside,
                theta_prior=prior_jax,
                training_overrides={
                    "learning_rate": 1e-4,
                    "min_lr_ratio": 1.0,
                },
                reference_theta=theta_0,
                multiround_overrides={
                    "prng_integer_seed": args.ssnle_seed,
                    "plot_save_dir": args.out_dir,
                    "simulation_budget": args.n_simulations,
                    "n_rounds": args.n_rounds,
                    "likelihood_chunk_size_gb": 0.5,
                    "n_likelihood_samples": 10_000,
                },
                flow_overrides={
                    "decoder_n_neurons": 128,
                    "decoder_n_layers": 4,
                    "architecture": 4 * ["MAF"] + ["surjective_MAF"] + 6 * ["MAF"],
                    "data_reduction_factor": 0.5,
                },
                data_spec=data_spec,
            )
        else:
            raise KeyError(f"Mode {mode} not recognised.")

        def model_sim_wrapper(
            npkey: NPKey,
            params: dict[str, np.ndarray],
            noise: bool = True,
            ui: MultiRoundInfererUI | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
            return batch_simulate(
                params,
                simulator_wrapper,
                n_workers=args.n_workers,
                ui=ui,
                rng_key=npkey,
            )

        inferer = MultiRoundInferer(
            mode,
            prior,
            model_sim_wrapper,
            (x0, mask),
            multi_round_config=scenario.multiround,
            transform_config=scenario.transforms,
            nflow_config=scenario.flow,
            train_config=scenario.training,
            use_ui=not args.no_ui,
        )
        inferer.run()
