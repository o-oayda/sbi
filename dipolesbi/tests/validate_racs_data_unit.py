from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from dipolesbi.pipelines.validate_racs_data import (
    DataValidationError,
    validate_racs_data,
    write_report,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_files(tmp_path: Path):
    catalogue = tmp_path / "catalogue.fits"
    catalogue.write_bytes(b"catalogue")
    paf_root = tmp_path / "paf"
    paf_file = paf_root / "mid1" / "ak01.csv"
    paf_file.parent.mkdir(parents=True)
    paf_file.write_bytes(b"paf temperatures")

    manifest = tmp_path / "paf-manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "paf",
                "file_glob": "mid1/ak*.csv",
                "files": [
                    {
                        "relative_path": "mid1/ak01.csv",
                        "sha256": _sha256(paf_file),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "datasets.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "catalogue": {
                        "type": "file",
                        "sha256": _sha256(catalogue),
                    },
                    "paf": {
                        "type": "file_collection",
                        "manifest": str(manifest),
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return registry, catalogue, paf_root, paf_file, manifest


def _validate(tmp_path: Path):
    registry, catalogue, paf_root, _, manifest = _dataset_files(tmp_path)
    return validate_racs_data(
        registry_path=registry,
        catalogue_id="catalogue",
        catalogue_path=catalogue,
        paf_id="paf",
        paf_root=paf_root,
        paf_manifest_path=manifest,
    )


def test_validate_racs_data_reports_verified_files(tmp_path):
    report = _validate(tmp_path)

    assert report["result"] == "valid"
    assert report["datasets"]["catalogue"]["valid"] is True
    paf = report["datasets"]["paf"]
    assert paf["valid"] is True
    assert paf["files"][0]["relative_path"] == "mid1/ak01.csv"
    assert paf["files"][0]["actual_sha256"] == paf["files"][0]["expected_sha256"]


def test_validate_racs_data_rejects_changed_catalogue(tmp_path):
    registry, catalogue, paf_root, _, manifest = _dataset_files(tmp_path)
    catalogue.write_bytes(b"changed")

    with pytest.raises(DataValidationError, match="checksum mismatch: catalogue"):
        validate_racs_data(
            registry_path=registry,
            catalogue_id="catalogue",
            catalogue_path=catalogue,
            paf_id="paf",
            paf_root=paf_root,
            paf_manifest_path=manifest,
        )


def test_validate_racs_data_rejects_changed_paf_file(tmp_path):
    registry, catalogue, paf_root, paf_file, manifest = _dataset_files(tmp_path)
    paf_file.write_bytes(b"changed")

    with pytest.raises(DataValidationError, match="checksum mismatch: paf/mid1/ak01"):
        validate_racs_data(
            registry_path=registry,
            catalogue_id="catalogue",
            catalogue_path=catalogue,
            paf_id="paf",
            paf_root=paf_root,
            paf_manifest_path=manifest,
        )


def test_validate_racs_data_rejects_unexpected_paf_file(tmp_path):
    registry, catalogue, paf_root, _, manifest = _dataset_files(tmp_path)
    (paf_root / "mid1" / "ak02.csv").write_bytes(b"unexpected")

    with pytest.raises(DataValidationError, match="unexpected file"):
        validate_racs_data(
            registry_path=registry,
            catalogue_id="catalogue",
            catalogue_path=catalogue,
            paf_id="paf",
            paf_root=paf_root,
            paf_manifest_path=manifest,
        )


def test_write_report_creates_parseable_yaml(tmp_path):
    report = _validate(tmp_path)
    output = tmp_path / "nested" / "validation.yaml"

    write_report(output, report)

    assert yaml.safe_load(output.read_text(encoding="utf-8")) == report
