"""Archive tests use tiny synthetic files, never catalogue data or inference."""

import hashlib
from copy import deepcopy
import zipfile

import numpy as np
import pytest
import yaml

from dipolesbi.pipelines.package_catwise_analysis import (
    CatwisePackagingError,
    package_catwise_analysis,
    sha256,
    verify_catwise_archive,
)


@pytest.fixture
def analysis(tmp_path, monkeypatch):
    root, result, observation = [
        tmp_path / name for name in ("repo", "result", "obs")
    ]
    for directory in (root, result, observation):
        directory.mkdir()
    for name in (
        "pyproject.toml",
        "uv.lock",
        "experiment.yaml",
        "observation.yaml",
        "model.yaml",
        "priors.yaml",
        "inference.yaml",
        "site.yaml",
    ):
        (root / name).write_text("test: true\n")
    configs = [
        root / name
        for name in (
            "experiment.yaml",
            "observation.yaml",
            "model.yaml",
            "priors.yaml",
            "inference.yaml",
            "site.yaml",
        )
    ]
    run = {
        "experiment_id": "tiny-s21",
        "execution": {"n_rounds": 2},
        "inference": {
            "multiround": {
                "plot_save_dir": str(result),
                "simulation_path": None,
                "load_simulations": None,
            }
        },
        "observation": {"catalogue_sha256": "a" * 64, "mask_sha256": "b" * 64},
        "paths": {
            "s21_catalogue": "/unavailable/catalogue.fits",
            "s21_mask": "/unavailable/mask.npy",
        },
        "provenance": {
            "config_paths": list(map(str, configs)),
            "config_sha256": {str(path): sha256(path) for path in configs},
        },
    }
    for name in (
        "samples_rnd-1.csv",
        "epoch_lnZ.npy",
        "nflow_checkpoint_r1.npz",
        "reference_observation.npz",
        "prepared_observation.npz",
        "native_mask.npy",
        "model_config.json",
        "configs.txt",
        "run_command.txt",
    ):
        (result / name).write_bytes(b"tiny scientific payload\n")
    (result / "experiment.resolved.yaml").write_text(yaml.safe_dump(run))
    report = {
        "simulator_native_mask_matches": True,
        "input_validation": {
            "result": "valid",
            "files": {
                "catalogue": {
                    "path": run["paths"]["s21_catalogue"],
                    "sha256": "a" * 64,
                },
                "mask": {"path": run["paths"]["s21_mask"], "sha256": "b" * 64},
            },
        },
    }
    (result / "validation-report.yaml").write_text(yaml.safe_dump(report))
    for name in (
        "reference_observation.npz",
        "reference_observation_native.npz",
        "native_mask.npy",
    ):
        (observation / name).write_bytes(b"tiny scientific payload\n")
    (result / "reference_observation_native.npz").write_bytes(
        (observation / "reference_observation_native.npz").read_bytes()
    )
    np.savez_compressed(
        result / "reference_observation.npz",
        x0=[1.0, np.nan],
        mask=[True, False],
    )
    np.savez_compressed(
        result / "prepared_observation.npz",
        x0=[1.0, np.nan],
        mask=[True, False],
        metadata_json='{"format_version": 1}',
    )
    (observation / "reference_observation.npz").write_bytes(
        (result / "prepared_observation.npz").read_bytes()
    )
    state = {
        "commit": "abc123",
        "dirty": True,
        "status_porcelain": " M model.py",
        "reproducible": False,
        "pyproject_sha256": sha256(root / "pyproject.toml"),
        "uv_lock_sha256": sha256(root / "uv.lock"),
    }
    (result / "software-state.yaml").write_text(yaml.safe_dump(state))
    monkeypatch.setattr(
        "dipolesbi.pipelines.package_catwise_analysis.capture_software_state",
        lambda root: {**state, "commit": "newer-packaging-commit"},
    )
    return run, {
        "result_dir": result,
        "observation_dir": observation,
        "repository_root": root,
        "output_dir": tmp_path / "artifacts" / "tiny-s21",
    }


def test_archive_records_products_science_and_checksums(analysis):
    run, kwargs = analysis
    archive = package_catwise_analysis(run, **kwargs)
    manifest = verify_catwise_archive(
        archive, expected_experiment_id="tiny-s21"
    )
    assert manifest["source"]["dirty"] is True
    assert manifest["source"]["commit"] == "abc123"
    assert manifest["packaging_source"]["commit"] == "newer-packaging-commit"
    names = {entry["archive_path"] for entry in manifest["files"]}
    assert "results/nflow_checkpoint_r1.npz" in names
    assert "results/reference_observation_native.npz" in names
    assert "observation/reference_observation_native.npz" in names
    assert len([name for name in names if name.startswith("configs/")]) == 6
    assert (
        archive.with_suffix(".zip.sha256").read_text()
        == f"{sha256(archive)}  {archive.name}\n"
    )
    with zipfile.ZipFile(archive) as bundle:
        for entry in manifest["files"]:
            assert (
                hashlib.sha256(bundle.read(entry["archive_path"])).hexdigest()
                == entry["sha256"]
            )
        assert all("catalogue.fits" not in name for name in bundle.namelist())


def test_missing_final_checkpoint_fails_before_writing(analysis):
    run, kwargs = analysis
    (kwargs["result_dir"] / "nflow_checkpoint_r1.npz").unlink()
    with pytest.raises(
        CatwisePackagingError, match="Missing required.*checkpoint"
    ):
        package_catwise_analysis(run, **kwargs)
    assert not kwargs["output_dir"].exists()


def test_tampered_member_fails_verification(analysis):
    run, kwargs = analysis
    archive = package_catwise_analysis(run, **kwargs)
    with zipfile.ZipFile(archive) as bundle:
        contents = {name: bundle.read(name) for name in bundle.namelist()}
    contents["results/samples_rnd-1.csv"] = b"altered samples"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, data in contents.items():
            bundle.writestr(name, data)
    with pytest.raises(CatwisePackagingError, match="Checksum mismatch"):
        verify_catwise_archive(archive)


def test_changed_prepared_input_rejected(analysis):
    run, kwargs = analysis
    (kwargs["observation_dir"] / "native_mask.npy").write_bytes(b"changed")
    with pytest.raises(CatwisePackagingError, match="changed after inference"):
        package_catwise_analysis(run, **kwargs)


def test_changed_native_observation_rejected(analysis):
    run, kwargs = analysis
    (
        kwargs["observation_dir"] / "reference_observation_native.npz"
    ).write_bytes(b"changed native observation")
    with pytest.raises(
        CatwisePackagingError,
        match="changed after inference: .*reference_observation_native",
    ):
        package_catwise_analysis(run, **kwargs)
    assert not kwargs["output_dir"].exists()


def test_missing_inference_native_observation_rejected(analysis):
    run, kwargs = analysis
    (kwargs["result_dir"] / "reference_observation_native.npz").unlink()
    with pytest.raises(
        CatwisePackagingError,
        match="Missing required.*reference_observation_native",
    ):
        package_catwise_analysis(run, **kwargs)


def test_stale_resolved_config_rejected(analysis):
    run, kwargs = analysis
    (kwargs["result_dir"] / "experiment.resolved.yaml").write_text(
        "experiment_id: old\n"
    )
    with pytest.raises(CatwisePackagingError, match="Saved resolved config"):
        package_catwise_analysis(run, **kwargs)


def test_inference_used_different_observation_rejected(analysis):
    run, kwargs = analysis
    np.savez_compressed(
        kwargs["result_dir"] / "reference_observation.npz",
        x0=[9.0, np.nan],
        mask=[True, False],
    )
    with pytest.raises(
        CatwisePackagingError, match="Inference reference x0 differs"
    ):
        package_catwise_analysis(run, **kwargs)


def test_runtime_output_override_can_be_packaged(analysis):
    run, kwargs = analysis
    loaded = deepcopy(run)
    loaded["inference"]["multiround"].update(
        plot_save_dir="/different/default/output",
        simulation_path="/different/default/output/resume/data",
        load_simulations="resume",
    )
    saved = deepcopy(run)
    saved["inference"]["multiround"].update(
        load_simulations="resume",
        simulation_path=str(
            kwargs["result_dir"].resolve() / "resume" / "data"
        ),
    )
    (kwargs["result_dir"] / "experiment.resolved.yaml").write_text(
        yaml.safe_dump(saved)
    )
    verify_catwise_archive(package_catwise_analysis(loaded, **kwargs))
    assert (
        loaded["inference"]["multiround"]["plot_save_dir"]
        == "/different/default/output"
    )


def test_changed_inference_environment_rejected(analysis):
    run, kwargs = analysis
    (kwargs["repository_root"] / "uv.lock").write_text("changed lock\n")
    with pytest.raises(
        CatwisePackagingError,
        match="Environment file changed after inference: uv.lock",
    ):
        package_catwise_analysis(run, **kwargs)


def test_missing_inference_software_state_rejected(analysis):
    run, kwargs = analysis
    (kwargs["result_dir"] / "software-state.yaml").unlink()
    with pytest.raises(
        CatwisePackagingError, match="Missing required.*software-state"
    ):
        package_catwise_analysis(run, **kwargs)


def test_archive_includes_inherited_config_sources(analysis):
    run, kwargs = analysis
    base = kwargs['repository_root'] / 'base.yaml'
    base.write_text('execution:\n  seed: 0\n')
    run['provenance']['config_paths'].append(str(base))
    run['provenance']['config_sha256'][str(base)] = sha256(base)
    (kwargs['result_dir'] / 'experiment.resolved.yaml').write_text(
        yaml.safe_dump(run)
    )
    archive = package_catwise_analysis(run, **kwargs)
    with zipfile.ZipFile(archive) as bundle:
        member = next(
            name for name in bundle.namelist() if name.endswith('-base.yaml')
        )
        assert bundle.read(member) == base.read_bytes()
    base.write_text('execution:\n  seed: 1\n')
    with pytest.raises(
        CatwisePackagingError, match='changed since resolution'
    ):
        package_catwise_analysis(run, **kwargs)
