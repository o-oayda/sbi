import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Union
import numpy as np
import jax
from dipolesbi.tools.configs import SimpleDipoleMapConfig
from dipolesbi.tools.jax_ns import JaxNestedSampler
from dipolesbi.tools.maps import SimpleDipoleMap, SimpleDipoleMapJax
from dipolesbi.tools.np_rngkey import npkey_from_jax
from dipolesbi.tools.priors_np import DipolePriorNP
import os


def load_reference_theta(config_text: str) -> Dict[str, Union[float, np.ndarray]]:
    """
    Extract the `reference_theta` dictionary from the experiment config file.
    """
    match = re.search(r"reference_theta=(\{.*?\})", config_text, flags=re.DOTALL)
    if match is None:
        raise ValueError("Could not find a reference_theta dict.")

    reference_theta_raw = match.group(1)

    try:
        reference_theta = eval(reference_theta_raw, {"array": np.array})
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Failed to parse reference_theta.") from exc

    processed_reference_theta: Dict[str, Union[float, np.ndarray]] = {}
    for key, value in reference_theta.items():
        if isinstance(value, np.ndarray):
            processed_reference_theta[key] = np.array(value, copy=True)
        else:
            processed_reference_theta[key] = value

    return processed_reference_theta


def load_model_nsides(config_text: str) -> Tuple[int, int]:
    """
    Extract `nside` and `downscale_nside` from the ModelConfig block.
    """
    match = re.search(
        r"SimpleDipoleMapConfig\([^)]*?nside=(\d+)[^)]*?downscale_nside=(\d+)",
        config_text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("Could not find SimpleDipoleMapConfig with nside values.")

    return int(match.group(1)), int(match.group(2))


def load_prng_seed(config_text: str) -> int:
    """
    Extract the `prng_integer_seed` from the MultiRoundInfererConfig block.
    """
    match = re.search(
        r"MultiRoundInfererConfig\([^)]*?prng_integer_seed=(\d+)",
        config_text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("Could not find prng_integer_seed.")

    return int(match.group(1))


def load_n_rounds(config_text: str) -> int:
    """
    Extract `n_rounds` from the MultiRoundInfererConfig block.
    """
    match = re.search(
        r"MultiRoundInfererConfig\([^)]*?n_rounds=(\d+)",
        config_text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("Could not find n_rounds.")

    return int(match.group(1))


if __name__ == "__main__":
    X0_PRNG_SEED = 42
    x0_rng_key = jax.random.PRNGKey(X0_PRNG_SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Directory containing samples_rnd-*.csv files.",
    )
    args = parser.parse_args()

    configs_path = args.experiment_dir / "configs.txt"
    config_text = configs_path.read_text()

    try:
        reference_theta = load_reference_theta(config_text)
    except ValueError as exc:
        raise ValueError(f"Failed to parse reference_theta in {configs_path}") from exc

    try:
        model_nside, model_downscale_nside = load_model_nsides(config_text)
    except ValueError as exc:
        raise ValueError(f"Failed to parse nside values in {configs_path}") from exc

    try:
        prng_integer_seed = load_prng_seed(config_text)
    except ValueError as exc:
        raise ValueError(f"Failed to parse prng_integer_seed in {configs_path}") from exc

    try:
        n_rounds = load_n_rounds(config_text)
    except ValueError as exc:
        raise ValueError(f"Failed to parse n_rounds in {configs_path}") from exc

    print(reference_theta)

    simpledipole_config = SimpleDipoleMapConfig(
        nside=model_nside, downscale_nside=model_downscale_nside
    )
    model = SimpleDipoleMap(simpledipole_config)
    model.catwise_mask()

    x0, coarse_mask = model.generate_dipole(
        npkey_from_jax(x0_rng_key), theta=reference_theta
    )
    native_dmap, native_mask = model.dmap_and_mask

    mean_density = reference_theta['mean_density']
    prior = DipolePriorNP(
        mean_count_range=[float(0.95*mean_density), float(1.05*mean_density)],
    )
    prior.change_kwarg('N', 'mean_density')
    prior_jax = prior.to_jax()
    adapter = prior_jax.get_adapter()

    true_model_jax = SimpleDipoleMapJax(
        nside=model_nside,
        downscale_nside=model_downscale_nside,
        reference_data=jax.device_put(x0).squeeze(), # coarse resolution
        reference_mask=jax.device_put(native_mask).squeeze() # native resolution
    )

    # scuffed way to derive the original key for that run
    current_key = jax.random.PRNGKey(prng_integer_seed)
    _, current_key = jax.random.split(current_key)
    for rnd in range(n_rounds):
        current_key, *_ = jax.random.split(current_key, 4)

    true_logl = true_model_jax.log_likelihood
    classic_jax_ns = JaxNestedSampler(true_logl, prior.to_jax())
    classic_jax_ns.setup(current_key, n_live=1000, n_delete=200)
    classic_nested_samples = classic_jax_ns.run()

    classic_nested_samples.to_csv(
        path_or_buf=os.path.join(args.experiment_dir, f'true_samples.csv')
    )
