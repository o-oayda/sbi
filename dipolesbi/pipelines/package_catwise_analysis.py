"""Package the completed, scoped S21 CatWISE NLE analysis without source catalogues."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Any, Mapping
import zipfile

import numpy as np
import yaml

from dipolesbi.pipelines.catwise_config import DEFAULT_EXPERIMENT, load_catwise_run


class CatwisePackagingError(ValueError):
    """A required analysis product is missing or inconsistent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_software_state(root: str | Path | None = None) -> dict[str, Any]:
    """Record the source state and environment identities at inference startup."""
    root = Path(root).expanduser().resolve() if root is not None else Path.cwd()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
        if result.returncode:
            raise CatwisePackagingError(
                result.stderr.strip() or "Cannot inspect Git provenance"
            )
        return result.stdout.rstrip("\n")

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": status,
        "reproducible": not bool(status),
        "pyproject_sha256": sha256(root / "pyproject.toml"),
        "uv_lock_sha256": sha256(root / "uv.lock"),
    }


def _payload_paths(
    run: Mapping[str, Any], result: Path, observation: Path, root: Path
) -> dict[str, Path]:
    last = run["execution"]["n_rounds"] - 1
    paths = {
        **{
            f"results/{name}": result / name
            for name in (
                f"samples_rnd-{last}.csv",
                "epoch_lnZ.npy",
                f"nflow_checkpoint_r{last}.npz",
                "reference_observation.npz",
                "reference_observation_native.npz",
                "prepared_observation.npz",
                "native_mask.npy",
                "model_config.json",
                "configs.txt",
                "run_command.txt",
                "experiment.resolved.yaml",
                "validation-report.yaml",
                "software-state.yaml",
            )
        },
        **{
            f"observation/{name}": observation / name
            for name in (
                "reference_observation_native.npz",
                "reference_observation.npz",
                "native_mask.npy",
            )
        },
        "software/pyproject.toml": root / "pyproject.toml",
        "software/uv.lock": root / "uv.lock",
    }
    for index, source in enumerate(run["provenance"]["config_paths"]):
        path = Path(source)
        paths[f"configs/{index:02d}-{path.name}"] = path
    for name, path in paths.items():
        if not path.is_file():
            raise CatwisePackagingError(
                f"Missing required analysis input {name}: {path}"
            )
    saved = yaml.safe_load((result / "experiment.resolved.yaml").read_text())
    expected_run = deepcopy(dict(run))
    multiround = expected_run["inference"]["multiround"]
    multiround["plot_save_dir"] = str(result.resolve())
    load_simulations = multiround.get("load_simulations")
    multiround["simulation_path"] = (
        str(result.resolve() / load_simulations / "data")
        if load_simulations is not None
        else None
    )
    if saved != expected_run:
        raise CatwisePackagingError(
            "Saved resolved config differs from the requested experiment/site"
        )
    for path, expected in run["provenance"]["config_sha256"].items():
        if sha256(Path(path)) != expected:
            raise CatwisePackagingError(
                f"Configuration changed since resolution: {path}"
            )
    report = yaml.safe_load((result / "validation-report.yaml").read_text())
    if (
        not isinstance(report, dict)
        or report.get("simulator_native_mask_matches") is not True
    ):
        raise CatwisePackagingError(
            "Validation report must confirm simulator native mask agreement"
        )
    report = report.get("input_validation")
    if not isinstance(report, dict) or report.get("result") != "valid":
        raise CatwisePackagingError(
            "Input validation report must declare result: valid"
        )
    for role, path_key, digest_key in (
        ("catalogue", "s21_catalogue", "catalogue_sha256"),
        ("mask", "s21_mask", "mask_sha256"),
    ):
        identity = report.get("files", {}).get(role, {})
        if (
            identity.get("sha256") != run["observation"][digest_key]
            or identity.get("path") != run["paths"][path_key]
        ):
            raise CatwisePackagingError(
                f"Validation report has incorrect {role} identity"
            )
    for first, second in (
        (
            result / "prepared_observation.npz",
            observation / "reference_observation.npz",
        ),
        (
            result / "reference_observation_native.npz",
            observation / "reference_observation_native.npz",
        ),
        (result / "native_mask.npy", observation / "native_mask.npy"),
    ):
        if sha256(first) != sha256(second):
            raise CatwisePackagingError(
                f"Prepared input changed after inference: {second}"
            )
    try:
        with (
            np.load(result / "reference_observation.npz", allow_pickle=False) as used,
            np.load(
                result / "prepared_observation.npz", allow_pickle=False
            ) as prepared,
        ):
            for field in ("x0", "mask"):
                if not np.array_equal(used[field], prepared[field], equal_nan=True):
                    raise CatwisePackagingError(
                        f"Inference reference {field} differs from prepared input"
                    )
            if "metadata_json" not in prepared.files:
                raise CatwisePackagingError("Prepared input is missing metadata_json")
    except (OSError, KeyError, ValueError) as error:
        if isinstance(error, CatwisePackagingError):
            raise
        raise CatwisePackagingError(
            f"Cannot validate inference reference observation: {error}"
        ) from error
    return paths


def package_catwise_analysis(
    run: Mapping[str, Any],
    *,
    result_dir: str | Path,
    observation_dir: str | Path,
    output_dir: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> Path:
    """Archive explicit workflow products and record hashes and Git state.

    Packaging checks recorded identities; it never opens or bundles the large
    source catalogue or hashgrid cache. ``output_dir`` is the archive directory.
    """
    experiment_id = run["experiment_id"]
    if not isinstance(experiment_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_id
    ):
        raise CatwisePackagingError("Invalid experiment ID")
    root = (
        Path(repository_root).expanduser().resolve() if repository_root else Path.cwd()
    )
    paths = _payload_paths(run, Path(result_dir), Path(observation_dir), root)
    source = yaml.safe_load((Path(result_dir) / "software-state.yaml").read_text())
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("commit"), str)
        or not isinstance(source.get("dirty"), bool)
        or not isinstance(source.get("status_porcelain"), str)
        or source.get("reproducible") is not (not source["dirty"])
    ):
        raise CatwisePackagingError("Invalid inference-time software state")
    for filename, key in (
        ("pyproject.toml", "pyproject_sha256"),
        ("uv.lock", "uv_lock_sha256"),
    ):
        if sha256(root / filename) != source.get(key):
            raise CatwisePackagingError(
                f"Environment file changed after inference: {filename}"
            )
    packaging_source = capture_software_state(root)
    payload = {name: path.read_bytes() for name, path in paths.items()}
    payload["software/git-state.yaml"] = yaml.safe_dump(source, sort_keys=True).encode()
    manifest = {
        "format_version": 1,
        "workflow": "catwise-s21-nle",
        "experiment_id": experiment_id,
        "source": source,
        "packaging_source": packaging_source,
        "files": [
            {
                "archive_path": name,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in sorted(payload.items())
        ],
    }
    payload["artifact-manifest.yaml"] = yaml.safe_dump(
        manifest, sort_keys=True
    ).encode()
    output = Path(output_dir) if output_dir else root / "artifacts" / experiment_id
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{experiment_id}.zip"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output, suffix=".zip", delete=False
        ) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as bundle:
            for name, data in sorted(payload.items()):
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                bundle.writestr(info, data)
        verify_catwise_archive(temporary, expected_experiment_id=experiment_id)
        os.replace(temporary, archive)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    archive.with_suffix(".zip.sha256").write_text(
        f"{sha256(archive)}  {archive.name}\n"
    )
    return archive


def verify_catwise_archive(
    path: str | Path, *, expected_experiment_id: str | None = None
) -> dict[str, Any]:
    """Check member safety, completeness and every recorded payload checksum."""
    try:
        with zipfile.ZipFile(path) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)):
                raise CatwisePackagingError("Duplicate archive members")
            if any(
                PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                for name in names
            ):
                raise CatwisePackagingError("Unsafe archive member")
            manifest = yaml.safe_load(bundle.read("artifact-manifest.yaml"))
            if (
                not isinstance(manifest, dict)
                or manifest.get("format_version") != 1
                or manifest.get("workflow") != "catwise-s21-nle"
            ):
                raise CatwisePackagingError("Invalid CatWISE artifact manifest")
            if (
                expected_experiment_id is not None
                and manifest.get("experiment_id") != expected_experiment_id
            ):
                raise CatwisePackagingError("Incorrect experiment ID")
            members = manifest["files"]
            declared = [entry["archive_path"] for entry in members]
            if len(declared) != len(set(declared)) or set(names) != set(declared) | {
                "artifact-manifest.yaml"
            }:
                raise CatwisePackagingError("Archive inventory differs from manifest")
            for entry in members:
                data = bundle.read(entry["archive_path"])
                if (
                    len(data) != entry["size"]
                    or hashlib.sha256(data).hexdigest() != entry["sha256"]
                ):
                    raise CatwisePackagingError(
                        f"Checksum mismatch: {entry['archive_path']}"
                    )
            return manifest
    except (OSError, KeyError, TypeError, zipfile.BadZipFile, yaml.YAMLError) as error:
        raise CatwisePackagingError(f"Invalid analysis archive: {error}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Source checkout for software and Git provenance (default: current directory)",
    )
    args = parser.parse_args(argv)
    try:
        run = load_catwise_run(args.experiment, args.site)
        archive = package_catwise_analysis(
            run,
            result_dir=args.result_dir,
            observation_dir=args.observation_dir,
            output_dir=args.output_dir,
            repository_root=args.repository_root,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
