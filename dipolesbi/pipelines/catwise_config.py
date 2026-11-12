"""Resolve the explicit scientific and site configuration for S21 CatWISE NLE.

Loading a configuration reads only YAML files. Catalogue and mask bytes must be
validated by the observation pipeline before preparing data or starting catsim.
This module deliberately imports neither catsim, JAX nor Snakemake.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import yaml

from dipolesbi.pipelines.experiment_config import (
    ExperimentConfigError,
    resolve_config,
)

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "workflow" / "schemas"
DEFAULT_EXPERIMENT = (
    Path(__file__).resolve().parents[2]
    / "workflow/configs/catwise/s21_nle_gaussian_poisson.yaml"
)

GLASS_DEFAULTS = {
    "position_model": "isotropic",
    "glass_dndz": "catsim.configs:default_glass_dndz",
    "glass_bias": "catsim.configs:default_glass_bias",
    "glass_shell_edges": [index * 0.1 for index in range(36)],
    "glass_lmax": 128,
    "glass_ncorr": 3,
    "glass_cosmology": {
        "H0": 67.4,
        "ombh2": 0.0224,
        "omch2": 0.120,
        "tau": 0.054,
        "As": 2e-9,
        "ns": 0.96,
    },
}


class CatwiseConfigError(ValueError):
    """An invalid or inconsistent CatWISE configuration."""


def _read_yaml(path: Path) -> tuple[dict[str, Any], str]:
    try:
        contents = path.read_bytes()
        value = yaml.safe_load(contents)
    except (OSError, yaml.YAMLError) as error:
        raise CatwiseConfigError(
            f"Cannot read configuration {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CatwiseConfigError(
            f"Configuration must be a YAML mapping: {path}"
        )
    return value, hashlib.sha256(contents).hexdigest()


def _validate(value: dict[str, Any], kind: str, path: Path) -> None:
    schema, _ = _read_yaml(SCHEMA_DIR / f"catwise-{kind}.schema.yaml")
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: "/".join(map(str, error.absolute_path)),
    )
    if errors:
        error = errors[0]
        field = ".".join(map(str, error.absolute_path)) or "<root>"
        raise CatwiseConfigError(f"{path}: {field}: {error.message}")

    def finite(item: Any, field: str) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise CatwiseConfigError(f"{path}: {field} must be finite")
        if isinstance(item, dict):
            for key, child in item.items():
                finite(child, f"{field}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                finite(child, f"{field}[{index}]")

    finite(value, kind)


def _resolve_path(value: str, directory: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else directory / path).resolve()


def _validate_science(run: dict[str, Any]) -> None:
    observation = run["observation"]
    if run["model"]["use_noecl_mask"] != (
        not observation["mask_north_ecliptic"]
    ):
        raise CatwiseConfigError(
            "model.use_noecl_mask disagrees with observation.mask_north_ecliptic"
        )
    if observation["w1_min"] >= observation["w1_max"]:
        raise CatwiseConfigError("observation.w1_max must exceed w1_min")
    if run["model"]["cat_w1_max"] < observation["w1_max"]:
        raise CatwiseConfigError(
            "model.cat_w1_max must cover observation.w1_max"
        )
    if run["model"]["cat_w12_min"] > observation["w12_min"]:
        raise CatwiseConfigError(
            "model.cat_w12_min must cover observation.w12_min"
        )

    expected = [
        ("N", "log10_n_initial_samples", "uniform"),
        ("etaWX", "w1_extra_error", "uniform"),
        ("D", "observer_speed", "uniform"),
        ("phi", "dipole_longitude", "uniform"),
        ("theta", "dipole_latitude", "polar"),
    ]
    clustering = run["model"]["simulate_clustering"]
    if clustering == "poisson":
        expected.append(("lambda_clus", "lambda_clus", "uniform"))
    expected_fixed = {} if clustering == "poisson" else {"lambda_clus": 0.0}
    if run["fixed_parameters"] != expected_fixed:
        raise CatwiseConfigError(
            "Disabling additional clustering requires fixed_parameters: {lambda_clus: 0.0}; "
            "Poisson clustering requires an empty fixed_parameters mapping"
        )
    if run["model"]["position_model"] == "glass":
        if run["execution"]["backend"] != "numpy":
            raise CatwiseConfigError(
                "GLASS positions require execution.backend: numpy"
            )
        if clustering is not None:
            raise CatwiseConfigError(
                "GLASS cannot be combined with additional Poisson clustering"
            )
        edges = run["model"]["glass_shell_edges"]
        if any(high <= low for low, high in zip(edges, edges[1:])):
            raise CatwiseConfigError(
                "glass_shell_edges must be strictly increasing"
            )
        if run["model"]["glass_lmax"] > 3 * observation["native_nside"] - 1:
            raise CatwiseConfigError(
                "glass_lmax must not exceed 3 * native_nside - 1"
            )
        if run["model"]["glass_cosmology"]["H0"] <= 0:
            raise CatwiseConfigError("glass_cosmology.H0 must be positive")
    actual = [
        (p["short_name"], p["simulator_kwarg"], p["dist_type"])
        for p in run["priors"]
    ]
    if actual != expected:
        raise CatwiseConfigError(
            "priors.parameters must follow N, etaWX, D, phi, theta "
            "and lambda_clus only when Poisson clustering is enabled, "
            "with their S21 simulator arguments and distributions"
        )
    kwargs = {p["simulator_kwarg"] for p in run["priors"]}
    if set(run["reference_theta"]) != kwargs:
        raise CatwiseConfigError(
            "reference_theta must contain exactly the sampled parameters"
        )
    if kwargs & set(run["fixed_parameters"]):
        raise CatwiseConfigError(
            "fixed_parameters cannot override sampled parameters"
        )
    for prior in run["priors"]:
        low, high = prior["low"], prior["high"]
        if low >= high:
            raise CatwiseConfigError(
                f"Prior {prior['short_name']} must have low < high"
            )
        value = run["reference_theta"][prior["simulator_kwarg"]]
        if not low <= value <= high:
            raise CatwiseConfigError(
                f"reference_theta outside prior {prior['short_name']}"
            )
        if prior["dist_type"] == "polar" and not -90 <= low < high <= 90:
            raise CatwiseConfigError(
                "Polar latitude prior must stay within [-90, 90]"
            )

    execution = run["execution"]
    if execution["requested_simulations"] < execution["n_rounds"]:
        raise CatwiseConfigError(
            "requested_simulations must allocate at least one per round"
        )
    inference = run["inference"]
    if (
        inference["multiround"]["jax_ns_n_delete"]
        >= inference["multiround"]["jax_ns_n_live"]
    ):
        raise CatwiseConfigError(
            "jax_ns_n_delete must be smaller than jax_ns_n_live"
        )
    if inference["data_transform"]["kind"] != "zscore":
        raise CatwiseConfigError(
            "S21 NLE requires the explicit zscore data transform"
        )
    if inference["theta_transform"]["kind"] != "dipole_cartesian":
        raise CatwiseConfigError(
            "S21 NLE requires the dipole_cartesian theta transform"
        )
    if (
        inference["training"]["min_n_iter"]
        > inference["training"]["max_n_iter"]
    ):
        raise CatwiseConfigError(
            "training.min_n_iter must not exceed max_n_iter"
        )
    flow = inference["flow"]
    if (
        flow["embed_target_transform_in_flow"]
        != inference["data_transform"]["embed_in_flow"]
    ):
        raise CatwiseConfigError(
            "flow.embed_target_transform_in_flow must match data_transform.embed_in_flow"
        )
    if flow["conditioner"] != "made":
        raise CatwiseConfigError(
            "The configured affine/rational-quadratic MAF requires MADE"
        )
    if "healpix_funnel" in flow["architecture"]:
        raise CatwiseConfigError(
            "This CatWISE summary vector does not support healpix_funnel"
        )
    if (
        "surjective_MAF" in flow["architecture"]
        and flow["data_reduction_factor"] is None
    ):
        raise CatwiseConfigError(
            "surjective_MAF requires data_reduction_factor"
        )
    if inference["multiround"]["dequantise_data"]:
        raise CatwiseConfigError(
            "The continuous native log_dispersion summary cannot be dequantised"
        )


def load_catwise_run(
    experiment_path: str | Path, site_path: str | Path
) -> dict[str, Any]:
    """Load, validate and fully resolve one S21 NLE run without reading data.

    Scientific layer filenames are relative to the experiment YAML. Site paths
    are relative to the site YAML. ``source_catalogue`` and ``hashgrid_cache``
    are optional; null preserves catsim's existing cache discovery behavior.
    Each layer may extend a sibling YAML by basename. Parent mappings are merged
    before validation; lists and scalar values are replaced by child values.
    The requested budget remains 50,000 for the representative run, while
    ``effective_simulations`` reports its 15 floor-divided rounds (49,995).
    """
    experiment_path = Path(experiment_path).expanduser().resolve()
    site_path = Path(site_path).expanduser().resolve()
    sources: dict[str, str] = {}
    config_layers: dict[str, list[str]] = {}

    def read(path: Path, kind: str) -> dict[str, Any]:
        try:
            value, inherited_paths = resolve_config(path, config_kind=kind)
        except OSError as error:
            raise CatwiseConfigError(
                f"Cannot read configuration {path}: {error}"
            ) from error
        except ExperimentConfigError as error:
            raise CatwiseConfigError(str(error)) from error
        if kind == "priors" and isinstance(value.get("parameters"), dict):
            parameters = []
            for name, entry in value["parameters"].items():
                if not isinstance(entry, dict) or (
                    "short_name" in entry and entry["short_name"] != name
                ):
                    raise CatwiseConfigError(
                        f"{path}: prior {name} needs a mapping with a matching name"
                    )
                parameters.append({**entry, "short_name": name})
            value["parameters"] = parameters
        _validate(value, kind, path)
        config_layers[kind] = [str(parent) for parent in inherited_paths]
        for parent in inherited_paths:
            _, digest = _read_yaml(parent)
            sources[str(parent)] = digest
        return value

    experiment = read(experiment_path, "experiment")
    site = read(site_path, "site")
    layers = {
        kind: read(
            _resolve_path(experiment[kind], experiment_path.parent), kind
        )
        for kind in ("observation", "model", "priors", "inference")
    }
    model = deepcopy(GLASS_DEFAULTS)
    supplied_model = layers["model"]
    model.update(supplied_model)
    model["glass_cosmology"] = {
        **GLASS_DEFAULTS["glass_cosmology"],
        **supplied_model.get("glass_cosmology", {}),
    }
    run = {
        "schema_version": experiment["schema_version"],
        "experiment_id": experiment["experiment_id"],
        "observation": layers["observation"],
        "model": model,
        "priors": layers["priors"]["parameters"],
        "reference_theta": layers["priors"]["reference_theta"],
        "fixed_parameters": layers["priors"]["fixed_parameters"],
        "inference": layers["inference"],
        "execution": deepcopy(experiment["execution"]),
        "paths": {
            key: (
                str(_resolve_path(value, site_path.parent))
                if value is not None
                else None
            )
            for key, value in {
                "source_catalogue": None,
                "hashgrid_cache": None,
                "glass_cache": None,
                **site["paths"],
            }.items()
        },
        "provenance": {
            "config_paths": list(sources),
            "config_sha256": sources,
            "config_layers": config_layers,
        },
    }
    _validate_science(run)
    execution = run["execution"]
    per_round = execution["requested_simulations"] // execution["n_rounds"]
    execution.update(
        simulations_per_round=per_round,
        effective_simulations=per_round * execution["n_rounds"],
    )
    run["inference"]["multiround"].update(
        simulation_budget=execution["requested_simulations"],
        n_rounds=execution["n_rounds"],
        prng_integer_seed=execution["seed"],
        reference_theta=deepcopy(run["reference_theta"]),
        plot_save_dir=str(
            Path(run["paths"]["output_root"])
            / "results"
            / run["experiment_id"]
        ),
        map_ndim=12 * run["observation"]["downscale_nside"] ** 2,
        summary_ndim=1,
        native_count_summary=run["observation"]["native_count_summary"],
        simulations_per_round=per_round,
        effective_simulations=execution["effective_simulations"],
        simulation_path=None,
    )
    multiround = run["inference"]["multiround"]
    if multiround["load_simulations"] is not None:
        multiround["simulation_path"] = str(
            Path(multiround["plot_save_dir"])
            / multiround["load_simulations"]
            / "data"
        )
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write resolved YAML instead of printing it",
    )
    args = parser.parse_args(argv)
    try:
        run = load_catwise_run(args.experiment, args.site)
    except CatwiseConfigError as error:
        parser.error(str(error))
    rendered = yaml.safe_dump(run, sort_keys=False)
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
