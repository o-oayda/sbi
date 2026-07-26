"""Build and verify a reproducible RACS analysis artefact."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Any
import zipfile

import yaml


ARTIFACT_FORMAT_VERSION = 1
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_EXPERIMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class AnalysisPackagingError(ValueError):
    """Raised when an analysis bundle cannot be built or verified."""


@dataclass(frozen=True)
class AnalysisInputs:
    """The workflow-connected files included in an analysis artefact."""

    final_posterior: Path
    reference_observation: Path
    experiment_config: Path
    observation_config: Path
    inference_config: Path
    data_validation_report: Path
    dataset_registry: Path
    paf_manifest: Path
    model_config: Path
    configs: Path
    run_command: Path
    pyproject: Path
    uv_lock: Path


_ARCHIVE_PATHS = {
    "final_posterior": "posterior/final-posterior.csv",
    "reference_observation": "observation/reference-observation.npz",
    "experiment_config": "configs/experiment.yaml",
    "observation_config": "configs/observation.yaml",
    "inference_config": "configs/inference.yaml",
    "data_validation_report": "provenance/data-validation-report.yaml",
    "dataset_registry": "provenance/dataset-registry.yaml",
    "paf_manifest": "provenance/paf-manifest.yaml",
    "model_config": "provenance/model_config.json",
    "configs": "provenance/configs.txt",
    "run_command": "provenance/run_command.txt",
    "pyproject": "software/pyproject.toml",
    "uv_lock": "software/uv.lock",
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repository_root: Path, *args: str, required: bool = True) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if result.returncode and required:
        detail = result.stderr.strip() or "git command failed"
        raise AnalysisPackagingError(detail)
    return value or None


def implementation_fingerprint(repository_root: Path) -> str:
    """Fingerprint committed package sources and the committed environment lock."""
    root = repository_root.resolve(strict=True)
    listing = _git(
        root,
        "ls-tree",
        "-r",
        "--full-tree",
        "HEAD",
        "--",
        "dipolesbi",
        "uv.lock",
    )
    if not listing:
        raise AnalysisPackagingError(
            "No committed dipolesbi sources or uv.lock were found in the repository."
        )
    return hashlib.sha256((listing + "\n").encode()).hexdigest()


def _repository_url(repository_root: Path) -> str | None:
    remote = _git(repository_root, "remote", "get-url", "origin", required=False)
    if remote and remote.startswith("git@github.com:"):
        return "https://github.com/" + remote.removeprefix("git@github.com:")
    return remote


def _load_mapping(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise AnalysisPackagingError(f"Invalid {description}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisPackagingError(f"{description} must contain a YAML mapping: {path}")
    return value


def _validated_inputs(experiment_id: str, inputs: AnalysisInputs) -> dict[str, Path]:
    if _EXPERIMENT_ID.fullmatch(experiment_id) is None:
        raise AnalysisPackagingError(f"Invalid experiment ID: {experiment_id!r}")

    paths: dict[str, Path] = {}
    for field in fields(inputs):
        path = Path(getattr(inputs, field.name)).expanduser().resolve(strict=True)
        if not path.is_file():
            raise AnalysisPackagingError(f"Analysis input is not a file: {path}")
        paths[field.name] = path

    experiment = _load_mapping(paths["experiment_config"], "experiment config")
    if experiment.get("experiment_id") != experiment_id:
        raise AnalysisPackagingError(
            f"Experiment config declares {experiment.get('experiment_id')!r}; "
            f"expected {experiment_id!r}."
        )
    args = experiment.get("args")
    n_rounds = args.get("n_rounds") if isinstance(args, Mapping) else None
    if not isinstance(n_rounds, int) or isinstance(n_rounds, bool) or n_rounds < 1:
        raise AnalysisPackagingError("Experiment config must declare args.n_rounds >= 1.")
    expected_sample_name = f"samples_rnd-{n_rounds - 1}.csv"
    if paths["final_posterior"].name != expected_sample_name:
        raise AnalysisPackagingError(
            f"Final posterior must be named {expected_sample_name!r}, got "
            f"{paths['final_posterior'].name!r}."
        )

    observation = _load_mapping(paths["observation_config"], "observation config")
    if observation.get("observation_id") != paths["observation_config"].stem:
        raise AnalysisPackagingError(
            "Observation ID must match the observation-config filename stem."
        )
    inference = _load_mapping(paths["inference_config"], "inference config")
    if inference.get("inference_id") != paths["inference_config"].stem:
        raise AnalysisPackagingError(
            "Inference ID must match the inference-config filename stem."
        )
    return paths


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, _ZIP_TIMESTAMP)
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _manifest(
    *,
    experiment_id: str,
    paths: Mapping[str, Path],
    repository_root: Path,
    fingerprint: str,
    dirty: bool,
) -> dict[str, Any]:
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "experiment_id": experiment_id,
        "source": {
            "repository": _repository_url(repository_root),
            "commit": _git(repository_root, "rev-parse", "HEAD"),
            "dirty": dirty,
            "reproducible": not dirty,
        },
        "implementation_fingerprint": fingerprint,
        "files": [
            {
                "role": role,
                "archive_path": _ARCHIVE_PATHS[role],
                "size": paths[role].stat().st_size,
                "sha256": sha256(paths[role]),
            }
            for role in sorted(paths, key=lambda item: _ARCHIVE_PATHS[item])
        ],
    }


def verify_analysis_archive(
    archive_path: str | Path,
    *,
    expected_experiment_id: str | None = None,
) -> dict[str, Any]:
    """Verify archive structure and every payload against its internal manifest."""
    archive = Path(archive_path)
    try:
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)):
                raise AnalysisPackagingError("Archive contains duplicate entries.")
            if "artifact-manifest.yaml" not in names:
                raise AnalysisPackagingError("Archive is missing artifact-manifest.yaml.")
            manifest = yaml.safe_load(bundle.read("artifact-manifest.yaml"))
            if not isinstance(manifest, dict):
                raise AnalysisPackagingError("Artifact manifest must be a YAML mapping.")
            if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
                raise AnalysisPackagingError("Unsupported artifact manifest version.")
            if (
                expected_experiment_id is not None
                and manifest.get("experiment_id") != expected_experiment_id
            ):
                raise AnalysisPackagingError(
                    "Artifact manifest contains the wrong experiment ID."
                )
            entries = manifest.get("files")
            if not isinstance(entries, list):
                raise AnalysisPackagingError("Artifact manifest files must be a list.")

            declared: set[str] = set()
            roles: set[str] = set()
            for entry in entries:
                if not isinstance(entry, dict):
                    raise AnalysisPackagingError("Invalid artifact manifest file entry.")
                role = entry.get("role")
                name = entry.get("archive_path")
                if role not in _ARCHIVE_PATHS or _ARCHIVE_PATHS[role] != name:
                    raise AnalysisPackagingError(f"Invalid file role or path: {role!r}")
                pure_path = PurePosixPath(name)
                if pure_path.is_absolute() or ".." in pure_path.parts:
                    raise AnalysisPackagingError(f"Unsafe archive path: {name!r}")
                if name in declared or role in roles:
                    raise AnalysisPackagingError("Artifact manifest contains duplicates.")
                declared.add(name)
                roles.add(role)
                try:
                    payload = bundle.read(name)
                except KeyError as error:
                    raise AnalysisPackagingError(
                        f"Archive is missing declared file: {name}"
                    ) from error
                if len(payload) != entry.get("size"):
                    raise AnalysisPackagingError(f"Size mismatch for {name}.")
                actual = hashlib.sha256(payload).hexdigest()
                if actual != entry.get("sha256"):
                    raise AnalysisPackagingError(f"Checksum mismatch for {name}.")

            if roles != set(_ARCHIVE_PATHS):
                missing = ", ".join(sorted(set(_ARCHIVE_PATHS) - roles))
                raise AnalysisPackagingError(f"Artifact manifest is missing roles: {missing}")
            if set(names) != declared | {"artifact-manifest.yaml"}:
                raise AnalysisPackagingError("Archive contains undeclared files.")
    except (OSError, zipfile.BadZipFile, yaml.YAMLError) as error:
        raise AnalysisPackagingError(f"Cannot verify archive {archive}: {error}") from error
    return manifest


def package_racs_analysis(
    *,
    experiment_id: str,
    inputs: AnalysisInputs,
    artifacts_root: str | Path,
    repository_root: str | Path,
    expected_implementation_fingerprint: str | None = None,
    allow_dirty: bool = False,
) -> tuple[Path, Path]:
    """Create, verify, and atomically publish an analysis ZIP and checksum."""
    repository = Path(repository_root).expanduser().resolve(strict=True)
    status = _git(repository, "status", "--porcelain", "--untracked-files=normal")
    dirty = bool(status)
    if dirty and not allow_dirty:
        raise AnalysisPackagingError(
            "Refusing to package a dirty Git worktree; commit changes or pass "
            "--allow-dirty for an explicitly unreproducible development bundle."
        )
    fingerprint = implementation_fingerprint(repository)
    if (
        expected_implementation_fingerprint is not None
        and fingerprint != expected_implementation_fingerprint
    ):
        raise AnalysisPackagingError(
            "Implementation fingerprint does not match the workflow parameter."
        )
    paths = _validated_inputs(experiment_id, inputs)
    manifest = _manifest(
        experiment_id=experiment_id,
        paths=paths,
        repository_root=repository,
        fingerprint=fingerprint,
        dirty=dirty,
    )
    manifest_bytes = yaml.safe_dump(
        manifest, sort_keys=False, allow_unicode=True
    ).encode("utf-8")

    output_dir = Path(artifacts_root).expanduser() / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{experiment_id}.analysis.zip"
    checksum = output_dir / f"{experiment_id}.analysis.zip.sha256"
    temporary_archive: Path | None = None
    temporary_checksum: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_dir,
            prefix=f".{archive.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_archive = Path(stream.name)
        with zipfile.ZipFile(
            temporary_archive,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as bundle:
            entries = [
                (_ARCHIVE_PATHS[role], paths[role].read_bytes()) for role in paths
            ]
            entries.append(("artifact-manifest.yaml", manifest_bytes))
            for archive_path, payload in sorted(entries):
                bundle.writestr(_zip_info(archive_path), payload)

        verify_analysis_archive(
            temporary_archive, expected_experiment_id=experiment_id
        )
        archive_digest = sha256(temporary_archive)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=f".{checksum.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_checksum = Path(stream.name)
            stream.write(f"{archive_digest}  {archive.name}\n")
        os.replace(temporary_archive, archive)
        temporary_archive = None
        os.replace(temporary_checksum, checksum)
        temporary_checksum = None
    finally:
        for temporary in (temporary_archive, temporary_checksum):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return archive, checksum


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--implementation-fingerprint")
    parser.add_argument("--allow-dirty", action="store_true")
    for field in fields(AnalysisInputs):
        parser.add_argument(
            "--" + field.name.replace("_", "-"), type=Path, required=True
        )
    return parser


def main() -> None:
    args = _parser().parse_args()
    inputs = AnalysisInputs(
        **{field.name: getattr(args, field.name) for field in fields(AnalysisInputs)}
    )
    try:
        archive, checksum = package_racs_analysis(
            experiment_id=args.experiment_id,
            inputs=inputs,
            artifacts_root=args.artifacts_root,
            repository_root=args.repository_root,
            expected_implementation_fingerprint=args.implementation_fingerprint,
            allow_dirty=args.allow_dirty,
        )
    except (AnalysisPackagingError, FileNotFoundError) as error:
        raise SystemExit(f"error: {error}") from None
    print(f"Created {archive}")
    print(f"Created {checksum}")


if __name__ == "__main__":
    main()
