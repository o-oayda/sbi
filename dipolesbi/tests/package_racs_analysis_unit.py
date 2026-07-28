from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import zipfile

import pytest
import yaml

import dipolesbi.pipelines.package_racs_analysis as packaging
from dipolesbi.pipelines.package_racs_analysis import (
    AnalysisInputs,
    AnalysisPackagingError,
    package_racs_analysis,
    verify_analysis_archive,
)


def _run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    (repository / "dipolesbi").mkdir(parents=True)
    (repository / "dipolesbi" / "module.py").write_text("VALUE = 1\n")
    (repository / "uv.lock").write_text("version = 1\n")
    _run("git", "init", "-q", cwd=repository)
    _run("git", "config", "user.email", "test@example.invalid", cwd=repository)
    _run("git", "config", "user.name", "Test User", cwd=repository)
    _run("git", "add", "dipolesbi/module.py", "uv.lock", cwd=repository)
    _run("git", "commit", "-qm", "fixture", cwd=repository)
    return repository


def _inputs(tmp_path: Path) -> AnalysisInputs:
    source = tmp_path / "inputs"
    source.mkdir()
    values = {
        "samples_rnd-1.csv": "a,b\n1,2\n",
        "reference-observation.npz": "observation",
        "reference-observation-native.npz": "native observation",
        "racs_example.yaml": yaml.safe_dump(
            {"experiment_id": "racs_example", "args": {"n_rounds": 2}}
        ),
        "observation_one.yaml": yaml.safe_dump(
            {"observation_id": "observation_one"}
        ),
        "inference_one.yaml": yaml.safe_dump({"inference_id": "inference_one"}),
        "validation-report.yaml": "result: valid\n",
        "datasets.yaml": "datasets: {}\n",
        "paf-manifest.yaml": "dataset_id: paf\n",
        "model_config.json": "{}\n",
        "configs.txt": "configuration\n",
        "run_command.txt": "dipolesbi-infer-racs\n",
        "pyproject.toml": "[project]\nname = 'fixture'\n",
        "uv.lock": "version = 1\n",
    }
    for name, content in values.items():
        (source / name).write_text(content)
    return AnalysisInputs(
        final_posterior=source / "samples_rnd-1.csv",
        reference_observation=source / "reference-observation.npz",
        native_reference_observation=source / "reference-observation-native.npz",
        experiment_config=source / "racs_example.yaml",
        observation_config=source / "observation_one.yaml",
        inference_config=source / "inference_one.yaml",
        data_validation_report=source / "validation-report.yaml",
        dataset_registry=source / "datasets.yaml",
        paf_manifest=source / "paf-manifest.yaml",
        model_config=source / "model_config.json",
        configs=source / "configs.txt",
        run_command=source / "run_command.txt",
        pyproject=source / "pyproject.toml",
        uv_lock=source / "uv.lock",
    )


def _package(tmp_path: Path, name: str = "artifacts") -> tuple[Path, Path]:
    return package_racs_analysis(
        experiment_id="racs_example",
        inputs=_inputs(tmp_path),
        artifacts_root=tmp_path / name,
        repository_root=_repository(tmp_path),
    )


def test_build_is_deterministic(tmp_path):
    repository = _repository(tmp_path)
    inputs = _inputs(tmp_path)
    first, _ = package_racs_analysis(
        experiment_id="racs_example",
        inputs=inputs,
        artifacts_root=tmp_path / "first",
        repository_root=repository,
    )
    second, _ = package_racs_analysis(
        experiment_id="racs_example",
        inputs=inputs,
        artifacts_root=tmp_path / "second",
        repository_root=repository,
    )

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()


def test_manifest_roles_sizes_and_hashes_match_extracted_files(tmp_path):
    archive, checksum = _package(tmp_path)
    manifest = verify_analysis_archive(
        archive, expected_experiment_id="racs_example"
    )

    assert {entry["role"] for entry in manifest["files"]} == set(
        packaging._ARCHIVE_PATHS
    )
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == sorted(bundle.namelist())
        for entry in manifest["files"]:
            payload = bundle.read(entry["archive_path"])
            assert len(payload) == entry["size"]
            assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
    digest, filename = checksum.read_text().split()
    assert digest == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert filename == archive.name


def test_missing_input_is_rejected(tmp_path):
    repository = _repository(tmp_path)
    inputs = _inputs(tmp_path)
    inputs.final_posterior.unlink()

    with pytest.raises(FileNotFoundError):
        package_racs_analysis(
            experiment_id="racs_example",
            inputs=inputs,
            artifacts_root=tmp_path / "artifacts",
            repository_root=repository,
        )


def test_wrong_experiment_id_is_rejected(tmp_path):
    repository = _repository(tmp_path)
    inputs = _inputs(tmp_path)

    with pytest.raises(AnalysisPackagingError, match="declares.*expected"):
        package_racs_analysis(
            experiment_id="another_experiment",
            inputs=inputs,
            artifacts_root=tmp_path / "artifacts",
            repository_root=repository,
        )


def test_wrong_final_round_sample_name_is_rejected(tmp_path):
    repository = _repository(tmp_path)
    inputs = _inputs(tmp_path)
    wrong = inputs.final_posterior.with_name("samples_rnd-0.csv")
    inputs.final_posterior.rename(wrong)
    inputs = AnalysisInputs(**{**inputs.__dict__, "final_posterior": wrong})

    with pytest.raises(AnalysisPackagingError, match="samples_rnd-1.csv"):
        package_racs_analysis(
            experiment_id="racs_example",
            inputs=inputs,
            artifacts_root=tmp_path / "artifacts",
            repository_root=repository,
        )


def test_corrupted_archive_is_rejected(tmp_path):
    archive, _ = _package(tmp_path)
    archive.write_bytes(archive.read_bytes()[:20])

    with pytest.raises(AnalysisPackagingError, match="Cannot verify archive"):
        verify_analysis_archive(archive)


def test_failed_verification_does_not_replace_outputs(tmp_path, monkeypatch):
    repository = _repository(tmp_path)
    inputs = _inputs(tmp_path)
    output_dir = tmp_path / "artifacts" / "racs_example"
    output_dir.mkdir(parents=True)
    archive = output_dir / "racs_example.analysis.zip"
    checksum = output_dir / "racs_example.analysis.zip.sha256"
    archive.write_bytes(b"existing archive")
    checksum.write_bytes(b"existing checksum")

    def fail_verification(*args, **kwargs):
        raise AnalysisPackagingError("injected verification failure")

    monkeypatch.setattr(packaging, "verify_analysis_archive", fail_verification)
    with pytest.raises(AnalysisPackagingError, match="injected"):
        package_racs_analysis(
            experiment_id="racs_example",
            inputs=inputs,
            artifacts_root=tmp_path / "artifacts",
            repository_root=repository,
        )

    assert archive.read_bytes() == b"existing archive"
    assert checksum.read_bytes() == b"existing checksum"
    assert not list(output_dir.glob("*.tmp"))


def test_dirty_worktree_requires_explicit_override(tmp_path):
    repository = _repository(tmp_path)
    inputs = _inputs(tmp_path)
    (repository / "dipolesbi" / "module.py").write_text("VALUE = 2\n")

    with pytest.raises(AnalysisPackagingError, match="dirty Git worktree"):
        package_racs_analysis(
            experiment_id="racs_example",
            inputs=inputs,
            artifacts_root=tmp_path / "refused",
            repository_root=repository,
        )

    archive, _ = package_racs_analysis(
        experiment_id="racs_example",
        inputs=inputs,
        artifacts_root=tmp_path / "allowed",
        repository_root=repository,
        allow_dirty=True,
    )
    manifest = verify_analysis_archive(archive)
    assert manifest["source"]["dirty"] is True
    assert manifest["source"]["reproducible"] is False
