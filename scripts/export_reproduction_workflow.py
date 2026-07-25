#!/usr/bin/env python3
"""Export one frozen experiment workflow into a reproduction repository."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import yaml


CORE_WORKFLOW_FILES = (
    "workflow/Snakefile",
    "workflow/configuration.smk",
    "workflow/schemas/racs-experiment.schema.yaml",
    "workflow/schemas/racs-observation.schema.yaml",
    "workflow/schemas/racs-inference.schema.yaml",
)
EXPERIMENT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
MANIFEST_PATH = Path("workflow/export-manifest.yaml")


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml_mapping(path: Path, description: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError(f"{description} must contain a YAML mapping: {path}")
    return values


def checked_reference(
    repository_root: Path,
    raw_path: Any,
    expected_directory: str,
    description: str,
) -> Path:
    """Resolve a config reference while keeping it inside its expected tree."""
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{description} must be a non-empty path string.")

    resolved = (repository_root / raw_path).resolve(strict=True)
    expected_root = (repository_root / expected_directory).resolve(strict=True)
    if not resolved.is_relative_to(expected_root) or not resolved.is_file():
        raise ValueError(
            f"{description} must refer to a file below {expected_directory}: {raw_path}"
        )
    return resolved.relative_to(repository_root)


def selected_files(repository_root: Path, experiment_name: str) -> list[Path]:
    """Resolve the workflow files required by one experiment."""
    if EXPERIMENT_NAME_PATTERN.fullmatch(experiment_name) is None:
        raise ValueError(f"Invalid experiment name: {experiment_name!r}")

    experiment_path = Path(
        f"workflow/configs/experiments/{experiment_name}.yaml"
    )
    absolute_experiment_path = repository_root / experiment_path
    if not absolute_experiment_path.is_file():
        raise FileNotFoundError(
            f"Experiment config does not exist: {absolute_experiment_path}"
        )

    experiment = load_yaml_mapping(absolute_experiment_path, "Experiment config")
    if experiment.get("experiment_id") != experiment_name:
        raise ValueError(
            f"Experiment filename selects {experiment_name!r}, but experiment_id is "
            f"{experiment.get('experiment_id')!r}."
        )

    observation_path = checked_reference(
        repository_root,
        experiment.get("observation_config"),
        "workflow/configs/observations",
        "observation_config",
    )
    inference_path = checked_reference(
        repository_root,
        experiment.get("inference_config"),
        "workflow/configs/inference",
        "inference_config",
    )

    paths = [
        *(Path(path) for path in CORE_WORKFLOW_FILES),
        experiment_path,
        observation_path,
        inference_path,
    ]
    for relative_path in paths:
        if not (repository_root / relative_path).is_file():
            raise FileNotFoundError(
                f"Required workflow file does not exist: {relative_path}"
            )
    return list(dict.fromkeys(paths))


def git_value(repository_root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def public_repository_url(remote: str | None) -> str | None:
    """Convert the common GitHub SSH remote form to a public HTTPS URL."""
    if remote is not None and remote.startswith("git@github.com:"):
        return "https://github.com/" + remote.removeprefix("git@github.com:")
    return remote


def build_manifest(
    repository_root: Path,
    experiment_name: str,
    relative_paths: list[Path],
) -> dict[str, Any]:
    status = git_value(repository_root, "status", "--porcelain")
    return {
        "format_version": 1,
        "experiment": experiment_name,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": public_repository_url(
                git_value(repository_root, "remote", "get-url", "origin")
            ),
            "commit": git_value(repository_root, "rev-parse", "HEAD"),
            "dirty": bool(status),
        },
        "files": {
            relative_path.as_posix(): {
                "sha256": sha256(repository_root / relative_path)
            }
            for relative_path in relative_paths
        },
    }


def differing_paths(
    repository_root: Path,
    destination_root: Path,
    relative_paths: list[Path],
) -> tuple[list[Path], list[Path]]:
    missing: list[Path] = []
    changed: list[Path] = []
    for relative_path in relative_paths:
        source = repository_root / relative_path
        destination = destination_root / relative_path
        if not destination.is_file():
            missing.append(relative_path)
        elif sha256(source) != sha256(destination):
            changed.append(relative_path)
    return missing, changed


def export_workflow(
    repository_root: Path,
    destination_root: Path,
    experiment_name: str,
    *,
    force: bool = False,
    check: bool = False,
) -> None:
    repository_root = repository_root.resolve(strict=True)
    destination_root = destination_root.expanduser().resolve()
    if destination_root == repository_root:
        raise ValueError("Destination must not be the dipolesbi repository itself.")
    if not destination_root.is_dir():
        raise FileNotFoundError(f"Destination directory does not exist: {destination_root}")

    relative_paths = selected_files(repository_root, experiment_name)
    missing, changed = differing_paths(
        repository_root,
        destination_root,
        relative_paths,
    )

    if check:
        if missing or changed:
            details = [
                *(f"missing: {path}" for path in missing),
                *(f"changed: {path}" for path in changed),
            ]
            raise RuntimeError("Exported workflow differs:\n" + "\n".join(details))
        print(f"Export matches {experiment_name}: {destination_root}")
        return

    if changed and not force:
        details = "\n".join(f"  {path}" for path in changed)
        raise FileExistsError(
            "Refusing to overwrite modified exported file(s):\n"
            f"{details}\nRe-run with --force after reviewing the differences."
        )

    for relative_path in relative_paths:
        source = repository_root / relative_path
        destination = destination_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    manifest = build_manifest(repository_root, experiment_name, relative_paths)
    manifest_destination = destination_root / MANIFEST_PATH
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_destination.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    print(
        f"Exported {len(relative_paths)} workflow files for {experiment_name} "
        f"to {destination_root}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one dipolesbi experiment workflow as a frozen copy."
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    try:
        export_workflow(
            repository_root,
            args.destination,
            args.experiment,
            force=args.force,
            check=args.check,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
