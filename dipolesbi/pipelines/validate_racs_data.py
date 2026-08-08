"""Validate the external datasets required by a RACS observation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml


class DataValidationError(ValueError):
    """Raised when a dataset does not match its declared identity."""


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml_mapping(path: Path, description: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise DataValidationError(
            f"{description} must contain a YAML mapping: {path}"
        )
    return values


def registry_dataset(
    registry: dict[str, Any], dataset_id: str, expected_type: str
) -> dict[str, Any]:
    datasets = registry.get("datasets")
    dataset = datasets.get(dataset_id) if isinstance(datasets, dict) else None
    if not isinstance(dataset, dict):
        raise DataValidationError(f"Dataset is absent from the registry: {dataset_id}")
    if dataset.get("type") != expected_type:
        raise DataValidationError(
            f"Dataset {dataset_id!r} has type {dataset.get('type')!r}; "
            f"expected {expected_type!r}."
        )
    return dataset


def validate_file_collection(
    *,
    dataset_id: str,
    root: Path,
    manifest_path: Path,
    registry: dict[str, Any],
    registry_path: Path,
    failures: list[str],
) -> dict[str, Any]:
    """Validate one manifest-backed collection and return its report entry."""
    root = root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    collection = registry_dataset(registry, dataset_id, "file_collection")
    expected_manifest = Path(collection["manifest"])
    if not expected_manifest.is_absolute():
        expected_manifest = registry_path.parent / expected_manifest
    if expected_manifest.resolve(strict=True) != manifest_path:
        raise DataValidationError(
            f"Registry manifest for {dataset_id!r} does not match the selected "
            f"manifest: {collection['manifest']}"
        )
    manifest = load_yaml_mapping(manifest_path, "File-collection manifest")
    if manifest.get("dataset_id") != dataset_id:
        raise DataValidationError(
            f"Manifest declares dataset_id {manifest.get('dataset_id')!r}; "
            f"expected {dataset_id!r}."
        )

    files = []
    declared_paths: set[Path] = set()
    for entry in manifest["files"]:
        relative_path = Path(entry["relative_path"])
        path = (root / relative_path).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise DataValidationError(
                f"Manifest entry is not a file below {root}: {relative_path}"
            )
        if path in declared_paths:
            raise DataValidationError(
                f"Manifest contains a duplicate file: {relative_path}"
            )
        declared_paths.add(path)

        actual = sha256(path)
        expected = entry["sha256"]
        if actual != expected:
            failures.append(f"checksum mismatch: {dataset_id}/{relative_path}")
        files.append(
            {
                "relative_path": relative_path.as_posix(),
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )

    matched_paths = {
        path.resolve(strict=True)
        for path in root.glob(manifest["file_glob"])
        if path.is_file()
    }
    unexpected = sorted(matched_paths - declared_paths)
    if unexpected:
        failures.extend(f"unexpected file: {path}" for path in unexpected)
    return {
        "type": "file_collection",
        "root": str(root),
        "manifest": str(manifest_path),
        "file_glob": manifest["file_glob"],
        "files": files,
    }


def validate_racs_data(
    *,
    registry_path: Path,
    catalogue_id: str,
    catalogue_path: Path,
    paf_id: str | None = None,
    paf_root: Path | None = None,
    paf_manifest_path: Path | None = None,
    noise_map_id: str | None = None,
    noise_map_root: Path | None = None,
    noise_map_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Validate file contents and return a provenance-ready report."""
    registry_path = registry_path.resolve(strict=True)
    catalogue_path = catalogue_path.resolve(strict=True)
    paf_values = (paf_id, paf_root, paf_manifest_path)
    if any(value is None for value in paf_values) and not all(
        value is None for value in paf_values
    ):
        raise DataValidationError(
            "PAF dataset ID, root, and manifest must be provided together."
        )
    noise_map_values = (noise_map_id, noise_map_root, noise_map_manifest_path)
    if any(value is None for value in noise_map_values) and not all(
        value is None for value in noise_map_values
    ):
        raise DataValidationError(
            "Noise-map dataset ID, root, and manifest must be provided together."
        )

    registry = load_yaml_mapping(registry_path, "Dataset registry")
    catalogue = registry_dataset(registry, catalogue_id, "file")
    failures: list[str] = []
    catalogue_expected = catalogue["sha256"]
    catalogue_actual = sha256(catalogue_path)
    if catalogue_actual != catalogue_expected:
        failures.append(f"checksum mismatch: {catalogue_id} ({catalogue_path})")

    datasets = {
        catalogue_id: {
            "type": "file",
            "path": str(catalogue_path),
            "expected_sha256": catalogue_expected,
            "actual_sha256": catalogue_actual,
        }
    }
    if paf_id is not None:
        assert paf_root is not None
        assert paf_manifest_path is not None
        datasets[paf_id] = validate_file_collection(
            dataset_id=paf_id,
            root=paf_root,
            manifest_path=paf_manifest_path,
            registry=registry,
            registry_path=registry_path,
            failures=failures,
        )
    if noise_map_id is not None:
        assert noise_map_root is not None
        assert noise_map_manifest_path is not None
        datasets[noise_map_id] = validate_file_collection(
            dataset_id=noise_map_id,
            root=noise_map_root,
            manifest_path=noise_map_manifest_path,
            registry=registry,
            registry_path=registry_path,
            failures=failures,
        )

    report = {
        "format_version": 1,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "valid" if not failures else "invalid",
        "datasets": datasets,
    }
    if failures:
        raise DataValidationError("Data validation failed:\n  " + "\n  ".join(failures))
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write a completed validation report atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        yaml.safe_dump(report, stream, sort_keys=False)
    os.replace(temporary_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the external datasets for one RACS observation."
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--catalogue-id", required=True)
    parser.add_argument("--catalogue-path", type=Path, required=True)
    parser.add_argument("--paf-id")
    parser.add_argument("--paf-root", type=Path)
    parser.add_argument("--paf-manifest", type=Path)
    parser.add_argument("--noise-map-id")
    parser.add_argument("--noise-map-root", type=Path)
    parser.add_argument("--noise-map-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = validate_racs_data(
            registry_path=args.registry,
            catalogue_id=args.catalogue_id,
            catalogue_path=args.catalogue_path,
            paf_id=args.paf_id,
            paf_root=args.paf_root,
            paf_manifest_path=args.paf_manifest,
            noise_map_id=args.noise_map_id,
            noise_map_root=args.noise_map_root,
            noise_map_manifest_path=args.noise_map_manifest,
        )
        write_report(args.output, report)
    except (DataValidationError, FileNotFoundError, KeyError) as error:
        raise SystemExit(f"error: {error}") from None


if __name__ == "__main__":
    main()
