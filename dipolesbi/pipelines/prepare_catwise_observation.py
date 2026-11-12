"""Validate the S21 inputs and prepare the CatWISE observation for NLE."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import healpy as hp
import numpy as np
from astropy.table import Table

from dipolesbi.pipelines.validate_racs_data import (
    DataValidationError,
    sha256,
    write_report,
)


def validate_catwise_inputs(run: Mapping[str, Any]) -> dict[str, Any]:
    """Check declared file identities before interpreting either input product."""
    observation = run["observation"]
    paths = run["paths"]
    if (
        observation["catalogue_version"] != "S21"
        or observation["base_mask_version"] != "S21"
    ):
        raise DataValidationError(
            "Only the S21 catalogue and S21 mask are supported."
        )
    if observation["native_nside"] != 64:
        raise DataValidationError("The S21 mask requires native_nside=64.")
    files = {}
    for name, path_key, digest_key in (
        ("catalogue", "s21_catalogue", "catalogue_sha256"),
        ("mask", "s21_mask", "mask_sha256"),
    ):
        path = Path(paths[path_key]).expanduser().resolve(strict=True)
        actual = sha256(path)
        expected = observation[digest_key]
        if actual != expected:
            raise DataValidationError(f"checksum mismatch for {name}: {path}")
        files[name] = {"path": str(path), "sha256": actual}

    table = Table.read(files["catalogue"]["path"], format="fits")
    missing = {"w1", "l", "b"} - set(table.colnames)
    if missing:
        raise DataValidationError(
            f"Catalogue is missing columns: {', '.join(sorted(missing))}"
        )
    for name in ("w1", "l", "b"):
        if not np.issubdtype(table[name].dtype, np.number):
            raise DataValidationError(
                f"Catalogue column {name} must be numeric."
            )
    mask = np.load(files["mask"]["path"], allow_pickle=False)
    if mask.shape != (hp.nside2npix(64),) or not np.all(
        (mask == 0) | (mask == 1)
    ):
        raise DataValidationError(
            "S21 mask must be a binary, one-dimensional nside=64 map."
        )
    return {
        "result": "valid",
        "catalogue_version": "S21",
        "base_mask_version": "S21",
        "mask_ordering": "RING",
        "mask_coordinates": "Galactic",
        "files": files,
    }


def build_native_retain_mask(
    run: Mapping[str, Any], *, report: Mapping[str, Any] | None = None
) -> np.ndarray:
    """Return a Galactic NEST retain mask, including the optional ecliptic disc."""
    if report is None:
        report = validate_catwise_inputs(run)
    mask = np.load(report["files"]["mask"]["path"], allow_pickle=False)
    retain = hp.reorder(mask, r2n=True).astype(bool)
    if run["observation"]["mask_north_ecliptic"]:
        from catsim.utils.healsphere import Mask

        retain[Mask(nside=64).north_ecliptic_mask()] = False
    return retain


def observation_metadata(
    run: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe all settings and identities that define the observation vector."""
    obs = run["observation"]
    keys = (
        "observation_id",
        "catalogue_version",
        "base_mask_version",
        "native_nside",
        "downscale_nside",
        "native_count_summary",
        "w1_min",
        "w1_max",
        "w12_min",
        "mask_north_ecliptic",
    )
    coarse_npix = hp.nside2npix(obs["downscale_nside"])
    extra = {}
    if obs.get("native_pseudo_cl", False):
        lo, hi = obs["pseudo_cl_range"]
        extra = {
            "pseudo_cl": {
                "feature_slice": [coarse_npix + 1, coarse_npix + 2 + hi - lo],
                "ell_min": lo,
                "ell_max": hi,
                "estimator": "NaMaster compute_coupled_cell",
                "shot_noise_subtracted": False,
                "decoupled": False,
            }
        }
    return {
        "format_version": 1,
        "observation": {key: obs[key] for key in keys},
        "catalogue_sha256": report["files"]["catalogue"]["sha256"],
        "mask_sha256": report["files"]["mask"]["sha256"],
        "native_ordering": "NEST",
        "coordinates": "Galactic",
        "feature_layout": {
            "coarse_nest_pixels": [0, coarse_npix],
            "log_dispersion": coarse_npix,
            **extra,
        },
        "selection": "preselected S21 catalogue; w1_min < w1 < w1_max; no colour recut",
    }


def _atomic_save(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer(stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_healpix_plot(path: str | Path, native_map: np.ndarray) -> Path:
    """Save a native Galactic NEST projection, matching the RACS plots."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hp.projview(native_map, nest=True)
    figure = plt.gcf()
    try:
        figure.savefig(path)
    finally:
        plt.close(figure)
    return path


def prepare_catwise_observation(
    run: Mapping[str, Any],
    output_dir: str | Path,
    *,
    mask_path: str | Path | None = None,
) -> dict[str, Path]:
    """Prepare native and hybrid observations from the verified S21 products."""
    from dipolesbi.pipelines.catwise_components import process_native_sample

    report = validate_catwise_inputs(run)
    obs = run["observation"]
    retain = build_native_retain_mask(run, report=report)
    output = Path(output_dir)
    if mask_path is not None:
        mask_path = Path(mask_path)
        if mask_path.resolve() != (output / "native_mask.npy").resolve():
            raise DataValidationError(
                "Supplied mask must be output_dir/native_mask.npy."
            )
        supplied_mask = np.load(mask_path, allow_pickle=False)
        if supplied_mask.dtype != np.bool_ or not np.array_equal(
            supplied_mask, retain
        ):
            raise DataValidationError(
                "Supplied native mask does not match the run."
            )
        retain = supplied_mask
    table = Table.read(report["files"]["catalogue"]["path"], format="fits")
    selected = (table["w1"] > obs["w1_min"]) & (table["w1"] < obs["w1_max"])
    pixels = hp.ang2pix(
        obs["native_nside"],
        table["l"][selected],
        table["b"][selected],
        lonlat=True,
        nest=True,
    )
    native = np.bincount(
        pixels, minlength=hp.nside2npix(obs["native_nside"])
    ).astype(np.float32)
    native[~retain] = np.nan
    x, mask = process_native_sample(native, retain, obs)
    metadata = json.dumps(observation_metadata(run, report), sort_keys=True)
    paths = {
        "native_mask": output / "native_mask.npy",
        "native": output / "reference_observation_native.npz",
        "processed": output / "reference_observation.npz",
        "native_plot": output / "reference_observation_native.png",
    }
    if mask_path is None:
        _atomic_save(
            paths["native_mask"],
            lambda f: np.save(f, retain, allow_pickle=False),
        )
    _atomic_save(
        paths["native"],
        lambda f: np.savez_compressed(
            f, x0=native, mask=retain, metadata_json=metadata
        ),
    )
    _atomic_save(
        paths["processed"],
        lambda f: np.savez_compressed(
            f, x0=x, mask=mask, metadata_json=metadata
        ),
    )
    save_healpix_plot(paths["native_plot"], native)
    return paths


def load_prepared_observation(
    run: Mapping[str, Any], path: str | Path
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load an observation only if its inputs, settings and native mask match."""
    path = Path(path)
    report = validate_catwise_inputs(run)
    expected = observation_metadata(run, report)
    with np.load(path, allow_pickle=False) as data:
        x = np.array(data["x0"], copy=True)
        mask = np.array(data["mask"], copy=True)
        metadata = json.loads(str(data["metadata_json"].item()))
    if metadata != expected:
        raise DataValidationError(
            "Prepared observation metadata does not match the run."
        )
    native_mask = np.load(path.parent / "native_mask.npy", allow_pickle=False)
    expected_mask = build_native_retain_mask(run, report=report)
    if native_mask.dtype != np.bool_ or not np.array_equal(
        native_mask, expected_mask
    ):
        raise DataValidationError(
            "Prepared native mask does not match the run."
        )
    nsummary = 1
    if run["observation"].get("native_pseudo_cl", False):
        lo, hi = run["observation"]["pseudo_cl_range"]
        nsummary += hi - lo + 1
    nfeatures = hp.nside2npix(run["observation"]["downscale_nside"]) + nsummary
    if (
        x.shape != (nfeatures,)
        or mask.shape != (nfeatures,)
        or mask.dtype != np.bool_
    ):
        raise DataValidationError(
            "Prepared observation has incompatible feature or mask shape."
        )
    from catsim.utils.healsphere import downgrade_ignore_nan

    _, coarse_mask = downgrade_ignore_nan(
        np.where(expected_mask, 1.0, np.nan),
        expected_mask,
        run["observation"]["downscale_nside"],
    )
    expected_processed_mask = np.concatenate(
        (np.asarray(coarse_mask, dtype=bool), np.ones(nsummary, dtype=bool))
    )
    if not np.array_equal(mask, expected_processed_mask):
        raise DataValidationError(
            "Prepared observation mask does not match the native mask."
        )
    if not np.all(np.isfinite(x[mask])) or not np.all(np.isnan(x[~mask])):
        raise DataValidationError(
            "Prepared observation contains invalid feature values."
        )
    return x, mask, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "mask", "prepare"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--experiment", type=Path, required=True)
        sub.add_argument("--site", type=Path, required=True)
        if command == "prepare":
            sub.add_argument("--output-dir", type=Path, required=True)
            sub.add_argument("--mask-path", type=Path)
        else:
            sub.add_argument(
                "--output",
                type=Path,
                required=True,
                help="Validation report or native mask file.",
            )
            if command == "mask":
                sub.add_argument(
                    "--plot-output",
                    type=Path,
                    help="Mask projection PNG (defaults to mask.png beside the mask).",
                )
    args = parser.parse_args()
    from dipolesbi.pipelines.catwise_config import load_catwise_run

    try:
        run = load_catwise_run(args.experiment, args.site)
        if args.command == "validate":
            write_report(args.output, validate_catwise_inputs(run))
        elif args.command == "mask":
            retain = build_native_retain_mask(run)
            _atomic_save(
                args.output, lambda f: np.save(f, retain, allow_pickle=False)
            )
            save_healpix_plot(
                args.plot_output or args.output.parent / "mask.png", retain
            )
        else:
            prepare_catwise_observation(
                run, args.output_dir, mask_path=args.mask_path
            )
    except (
        DataValidationError,
        FileNotFoundError,
        KeyError,
        ValueError,
    ) as error:
        raise SystemExit(f"error: {error}") from None


if __name__ == "__main__":
    main()
