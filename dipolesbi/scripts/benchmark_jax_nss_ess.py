from __future__ import annotations

import argparse
import csv
import os
import re
import time
from pathlib import Path

import blackjax
import jax
from jax import numpy as jnp
import numpy as np
from anesthetic import NestedSamples
from tqdm import tqdm

from dipolesbi.tools.np_rngkey import NPKey
from dipolesbi.tools.sbi_io import (
    get_lnlike_from_chkpt,
    get_prior_from_chkpt,
    get_x0_mask_from_json,
)


BENCHMARK_RESULTS: list[dict] = []
NESTED_SAMPLES_BY_SETTING: dict[tuple[int, int], NestedSamples] = {}
LAST_NESTED_SAMPLES: NestedSamples | None = None
PARAMETER_COLUMNS: list[str] = []


def _parameter_columns(samples: NestedSamples) -> list[str]:
    columns = samples.columns
    if hasattr(columns, "get_level_values"):
        names = list(columns.get_level_values(0))
    else:
        names = list(columns)
    return [
        name for name in names
        if name not in {"logL", "logL_birth", "nlive"}
    ]


def plot_nested_samples(
    samples: NestedSamples | None = None,
    columns: list[str] | None = None,
    kinds="default",
):
    """
    Convenience helper for interactive use after running this script with -i.
    """
    if samples is None:
        if LAST_NESTED_SAMPLES is None:
            raise ValueError("No nested samples are available yet.")
        samples = LAST_NESTED_SAMPLES
    if columns is None:
        columns = _parameter_columns(samples)
    if kinds == "default":
        return samples.plot_2d(columns)
    return samples.plot_2d(columns, kinds=kinds)


def _infer_round(checkpoint: Path) -> int:
    match = re.search(r"nflow_checkpoint_r(\d+)\.npz$", checkpoint.name)
    if match is None:
        raise ValueError(f"Could not infer round from checkpoint name: {checkpoint}")
    return int(match.group(1))


def _infer_posterior_draws(output_dir: Path, default: int) -> int:
    config_path = output_dir / "configs.txt"
    if not config_path.exists():
        return default
    text = config_path.read_text(encoding="utf-8")
    match = re.search(r"simulations_per_round=(\d+)", text)
    return int(match.group(1)) if match else default


def _nested_samples_from_dead(dead, prior) -> NestedSamples:
    columns = prior.simulator_kwargs
    data = jnp.vstack([dead.particles[key] for key in columns]).T
    return NestedSamples(
        data,
        logL=dead.loglikelihood,
        logL_birth=dead.loglikelihood_birth,
        columns=columns,
        labels=prior.prior_names,
        logzero=jnp.nan,
    )


def _proposal_duplicate_stats(
    samples: NestedSamples,
    n_draws: int,
    random_state: np.random.Generator,
) -> dict[str, float | int]:
    draws = samples.sample(n=n_draws, replace=True, random_state=random_state)
    param_cols = [
        col for col in samples.columns
        if col not in {"logL", "logL_birth", "nlive"}
    ]
    theta = draws[param_cols].to_numpy(dtype=np.float32)
    unique, counts = np.unique(theta, axis=0, return_counts=True)
    duplicate_draws = int(n_draws - unique.shape[0])
    return {
        "posterior_draws": int(n_draws),
        "unique_theta": int(unique.shape[0]),
        "duplicate_draws": duplicate_draws,
        "duplicate_fraction": duplicate_draws / float(n_draws),
        "max_multiplicity": int(counts.max()),
        "repeated_theta": int(np.count_nonzero(counts > 1)),
    }


def _run_nss(
    *,
    lnlike,
    prior,
    rng_key,
    n_live: int,
    n_delete: int,
    inner_steps_mult: float,
    stop_delta_logz: float,
    max_steps: int | None,
) -> tuple[NestedSamples, int]:
    prior_key, run_key = jax.random.split(rng_key)
    particles = prior.get_initial_live_samples(prior_key, n_live)
    num_inner_steps = max(1, int(round(prior.ndim * inner_steps_mult)))
    algorithm = blackjax.nss(
        logprior_fn=prior.log_prob,
        loglikelihood_fn=lnlike,
        num_delete=n_delete,
        num_inner_steps=num_inner_steps,
    )
    init_fn = jax.jit(algorithm.init)
    step_fn = jax.jit(algorithm.step)

    live = init_fn(particles)
    dead = []
    steps = 0

    with tqdm(
        desc=f"n_live={n_live} n_delete={n_delete}",
        unit="step",
        leave=False,
    ) as pbar:
        while not live.logZ_live - live.logZ < stop_delta_logz:
            if max_steps is not None and steps >= max_steps:
                break
            run_key, subkey = jax.random.split(run_key)
            live, dead_info = step_fn(subkey, live)
            dead.append(dead_info)
            steps += 1
            pbar.update(1)

    final_dead = blackjax.ns.utils.finalise(live, dead)
    return _nested_samples_from_dead(final_dead, prior), steps


def _write_row(path: Path, fieldnames: list[str], row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> tuple[list[dict], dict[tuple[int, int], NestedSamples]]:
    global BENCHMARK_RESULTS, NESTED_SAMPLES_BY_SETTING, LAST_NESTED_SAMPLES
    global PARAMETER_COLUMNS

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark BlackJAX NSS ESS and posterior-resampling duplicates "
            "from an existing NLE checkpoint."
        )
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Run output directory containing reference_observation.npz.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Checkpoint round. Defaults to the latest nflow_checkpoint_r*.npz.",
    )
    parser.add_argument(
        "--n-live",
        type=int,
        nargs="+",
        default=[2000, 4000],
        help="Live-point counts to benchmark.",
    )
    parser.add_argument(
        "--delete-fracs",
        type=float,
        nargs="+",
        default=[i / 10 for i in range(1, 10)],
        help="Fractions of n_live to use for n_delete.",
    )
    parser.add_argument(
        "--inner-steps-mult",
        type=float,
        default=5.0,
        help="Use round(ndim * this) NSS inner slice steps.",
    )
    parser.add_argument(
        "--posterior-draws",
        type=int,
        default=None,
        help="Number of replacement posterior draws for duplicate diagnostics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base JAX/NumPy seed.",
    )
    parser.add_argument(
        "--stop-delta-logz",
        type=float,
        default=-3.0,
        help="Stop when logZ_live - logZ < this value.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on NSS outer steps for smoke tests.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="CSV results path. Defaults inside output_dir.",
    )
    parser.add_argument(
        "--save-nested",
        action="store_true",
        help="Save each generated NestedSamples CSV next to the results.",
    )
    args = parser.parse_args()

    checkpoints = sorted(
        args.output_dir.glob("nflow_checkpoint_r*.npz"),
        key=_infer_round,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No nflow_checkpoint_r*.npz in {args.output_dir}")
    checkpoint = (
        args.output_dir / f"nflow_checkpoint_r{args.round}.npz"
        if args.round is not None
        else checkpoints[-1]
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    round_id = _infer_round(checkpoint)
    posterior_draws = (
        args.posterior_draws
        if args.posterior_draws is not None
        else _infer_posterior_draws(args.output_dir, default=13_333)
    )
    out_csv = (
        args.out_csv
        if args.out_csv is not None
        else args.output_dir / f"jax_nss_ess_benchmark_r{round_id}.csv"
    )

    x0, mask = get_x0_mask_from_json(str(args.output_dir))
    lnlike = get_lnlike_from_chkpt(
        str(checkpoint),
        np.atleast_2d(x0),
        np.atleast_2d(mask),
    )
    prior = get_prior_from_chkpt(str(checkpoint))
    PARAMETER_COLUMNS = list(prior.simulator_kwargs)

    fieldnames = [
        "round",
        "checkpoint",
        "n_live",
        "n_delete",
        "delete_frac",
        "num_inner_steps",
        "steps",
        "rows",
        "neff",
        "n_positive_weights",
        "max_weight_frac",
        "logZ",
        "logZerr",
        "posterior_draws",
        "unique_theta",
        "duplicate_draws",
        "duplicate_fraction",
        "max_multiplicity",
        "repeated_theta",
        "seconds",
    ]

    results: list[dict] = []
    nested_samples_by_setting: dict[tuple[int, int], NestedSamples] = {}

    print(f"checkpoint: {checkpoint}")
    print(f"posterior_draws: {posterior_draws}")
    print(f"writing: {out_csv}")

    for n_live in args.n_live:
        for delete_frac in args.delete_fracs:
            n_delete = max(1, int(round(n_live * delete_frac)))
            n_delete = min(n_delete, n_live - 1)
            run_seed = [args.seed, round_id, n_live, n_delete]
            rng_key = jax.random.PRNGKey(args.seed)
            rng_key = jax.random.fold_in(rng_key, round_id)
            rng_key = jax.random.fold_in(rng_key, n_live)
            rng_key = jax.random.fold_in(rng_key, n_delete)

            start = time.perf_counter()
            samples, steps = _run_nss(
                lnlike=lnlike,
                prior=prior,
                rng_key=rng_key,
                n_live=n_live,
                n_delete=n_delete,
                inner_steps_mult=args.inner_steps_mult,
                stop_delta_logz=args.stop_delta_logz,
                max_steps=args.max_steps,
            )
            seconds = time.perf_counter() - start
            nested_samples_by_setting[(n_live, n_delete)] = samples
            LAST_NESTED_SAMPLES = samples

            weights = np.asarray(samples.get_weights(), dtype=float)
            weight_sum = weights.sum()
            dup_stats = _proposal_duplicate_stats(
                samples,
                posterior_draws,
                NPKey.from_seed(run_seed).generator(),
            )
            row = {
                "round": round_id,
                "checkpoint": str(checkpoint),
                "n_live": n_live,
                "n_delete": n_delete,
                "delete_frac": n_delete / float(n_live),
                "num_inner_steps": max(1, int(round(prior.ndim * args.inner_steps_mult))),
                "steps": steps,
                "rows": len(samples),
                "neff": float(samples.neff()),
                "n_positive_weights": int(np.count_nonzero(weights)),
                "max_weight_frac": float(weights.max() / weight_sum),
                "logZ": float(samples.logZ()),
                "logZerr": float(samples.logZ(100).std()),
                **dup_stats,
                "seconds": seconds,
            }
            _write_row(out_csv, fieldnames, row)
            results.append(row)
            print(
                f"n_live={n_live} n_delete={n_delete}: "
                f"neff={row['neff']:.1f}, "
                f"unique={row['unique_theta']}/{posterior_draws}, "
                f"dup_frac={row['duplicate_fraction']:.3f}, "
                f"logZ={row['logZ']:.2f}, "
                f"seconds={seconds:.1f}"
            )

            if args.save_nested:
                nested_path = out_csv.with_name(
                    f"{out_csv.stem}_nested_r{round_id}"
                    f"_nlive{n_live}_ndelete{n_delete}.csv"
                )
                samples.to_csv(nested_path)

    BENCHMARK_RESULTS = results
    NESTED_SAMPLES_BY_SETTING = nested_samples_by_setting
    return results, nested_samples_by_setting


if __name__ == "__main__":
    BENCHMARK_RESULTS, NESTED_SAMPLES_BY_SETTING = main()
