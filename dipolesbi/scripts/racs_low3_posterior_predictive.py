import argparse
import os
import re
from pathlib import Path

# platform_override = os.environ.get("DIPOLESBI_JAX_PLATFORMS")
# if platform_override is not None:
#     os.environ["JAX_PLATFORMS"] = platform_override

import jax
import numpy as np
import healpy as hp
from catsim import RacsLow3, RacsLow3Config

from dipolesbi.scripts.based_racs_low3 import (
    _build_real_sample,
    attach_native_generate_dipole,
    build_prior_and_reference_theta,
    build_scenario,
    make_model_sim_wrapper,
    make_simulator_wrapper,
)
from dipolesbi.tools.multiround_inferer import MultiRoundInferer


def _infer_round_id(checkpoint_path: Path) -> int:
    match = re.search(r"_r(\d+)\.npz$", checkpoint_path.name)
    if match is None:
        return 0
    return int(match.group(1))


def _infer_downscale_nside(
    checkpoint_path: Path,
    mode: str,
    native_nside: int,
) -> int | None:
    if mode != "NLE":
        return None

    with np.load(checkpoint_path, allow_pickle=True) as checkpoint:
        target_ndim = int(checkpoint["target_ndim"])

    trained_nside = hp.npix2nside(target_ndim)
    if trained_nside == native_nside:
        return None
    return trained_nside


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["NLE", "NPE"],
        help="Checkpoint mode to reload.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to an nflow_checkpoint_r*.npz file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path for the posterior-predictive mean map.",
    )
    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=10_000,
        help="Number of posterior samples to draw before predictive simulation.",
    )
    parser.add_argument(
        "--predictive-samples",
        type=int,
        default=500,
        help="Number of posterior predictive simulations to average.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for posterior and predictive sampling.",
    )
    parser.add_argument(
        "--predictive-native-resolution",
        action="store_true",
        help=(
            "Generate posterior predictive maps at native simulator resolution, "
            "even if the checkpoint was trained on downscaled data."
        ),
    )
    parser.add_argument(
        "--round-id",
        type=int,
        default=None,
        help="Optional round index override. Defaults to parsing from checkpoint name.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of local workers to use for predictive simulations.",
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
        "--downscale_nside",
        type=int,
        default=None,
        help=(
            "Optional observed-map nside override. "
            "If omitted, NLE checkpoints infer this from target_ndim."
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50_000,
        help="Chunk size used inside the simulator.",
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
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output_path = args.output
    if output_path is None:
        output_path = checkpoint.parent / "posterior_predictive_mean.npz"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    effective_downscale_nside = args.downscale_nside
    if effective_downscale_nside is None:
        effective_downscale_nside = _infer_downscale_nside(
            checkpoint_path=checkpoint,
            mode=args.mode,
            native_nside=args.nside,
        )

    config = RacsLow3Config(
        flux_min=args.flux_min,
        nside=args.nside,
        chunk_size=args.chunk_size,
        use_float32=False,
        downscale_nside=effective_downscale_nside,
        store_final_samples=True,
        alpha_mean=args.alpha_mean,
        alpha_sigma=args.alpha_sigma,
        fractional_error_flux_min_mjy=args.fractional_error_flux_min_mjy,
    )
    model = RacsLow3(config)
    model.initialise_data()

    x0, mask = _build_real_sample(model, flux_min=args.flux_min)
    effective_nside = int(np.sqrt(x0.size / 12))

    prior, theta_0, temp_pivot = build_prior_and_reference_theta(model)
    attach_native_generate_dipole(model)
    simulator_wrapper = make_simulator_wrapper(model, temp_pivot)
    model_sim_wrapper = make_model_sim_wrapper(
        simulator_wrapper=simulator_wrapper,
        n_workers=args.n_workers,
    )

    scenario = build_scenario(
        mode=args.mode,
        effective_nside=effective_nside,
        prior=prior,
        theta_0=theta_0,
        out_dir=str(output_path.parent),
        ssnle_seed=args.seed,
        n_rounds=1,
        n_simulations=1,
    )

    inferer = MultiRoundInferer(
        args.mode,
        prior,
        model_sim_wrapper,
        (x0, mask),
        multi_round_config=scenario.multiround,
        transform_config=scenario.transforms,
        nflow_config=scenario.flow,
        train_config=scenario.training,
        use_ui=False,
        model_config=RacsLow3Config,
    )
    inferer.ui = None
    inferer.current_round = (
        args.round_id if args.round_id is not None else _infer_round_id(checkpoint)
    )
    inferer.rng_key = jax.random.PRNGKey(args.seed)
    inferer.load_nflow_checkpoint(str(checkpoint))

    posterior_key = jax.random.PRNGKey(args.seed)
    if args.mode == "NPE":
        inferer.current_posterior_samples = inferer._sample_posterior(
            posterior_key,
            args.posterior_samples,
        )
    else:
        inferer._compute_posterior(posterior_key)

    if args.predictive_native_resolution:
        predictive_simulator = make_model_sim_wrapper(
            simulator_wrapper=make_simulator_wrapper(
                model,
                temp_pivot,
                native_output=True,
            ),
            n_workers=args.n_workers,
        )
        inferer.simulator_function = predictive_simulator

    mean_map, mean_mask = inferer.posterior_predictive_mean(args.predictive_samples)

    np.savez_compressed(
        output_path,
        mean_map=np.asarray(mean_map, dtype=np.float32),
        mean_mask=np.asarray(mean_mask, dtype=np.bool_),
        reference_data=np.asarray(inferer.reference_data, dtype=np.float32),
        reference_mask=np.asarray(inferer.reference_mask, dtype=np.bool_),
        checkpoint=np.asarray(str(checkpoint)),
        mode=np.asarray(args.mode),
        posterior_samples=np.asarray(args.posterior_samples, dtype=np.int32),
        predictive_samples=np.asarray(args.predictive_samples, dtype=np.int32),
        seed=np.asarray(args.seed, dtype=np.int32),
        round_id=np.asarray(inferer.current_round, dtype=np.int32),
    )
    print(output_path)


if __name__ == "__main__":
    main()
