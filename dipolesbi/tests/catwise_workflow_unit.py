"""Run the real preparation DAG and check scoped workflow rerun behavior."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from astropy.table import Table
import yaml

from dipolesbi.pipelines.experiment_config import resolve_config

from dipolesbi.pipelines.catwise_config import (
    DEFAULT_EXPERIMENT,
    load_catwise_run,
)

ROOT = Path(__file__).resolve().parents[2]
SNAKEFILE = ROOT / "workflow/Snakefile.catwise"


def edit(path, update):
    value, _ = resolve_config(path)
    if isinstance(value.get("parameters"), dict):
        value["parameters"] = [
            {"short_name": key, **entry}
            for key, entry in value["parameters"].items()
        ]
    update(value)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


@pytest.fixture
def workflow_case(tmp_path):
    pytest.importorskip("snakemake")
    configs = tmp_path / "config with spaces"
    shutil.copytree(DEFAULT_EXPERIMENT.parent, configs)
    catalogue = tmp_path / "tiny catalogue.fits"
    Table(
        {
            "w1": [9.0, 9.1, 10.0, 10.0, 16.4],
            "l": [0.0, 0.0, 0.0, 90.0, 180.0],
            "b": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    ).write(catalogue)
    mask = tmp_path / "input mask.npy"
    retain = np.ones(12 * 64**2, dtype=np.int8)
    retain[17] = 0
    np.save(mask, retain)
    experiment = configs / DEFAULT_EXPERIMENT.name
    edit(
        experiment,
        lambda c: c["execution"].update(
            requested_simulations=4, n_rounds=2, jax_platforms="cpu"
        ),
    )
    edit(
        configs / "observation.yaml",
        lambda c: c.update(
            catalogue_sha256=hashlib.sha256(
                catalogue.read_bytes()
            ).hexdigest(),
            mask_sha256=hashlib.sha256(mask.read_bytes()).hexdigest(),
        ),
    )
    site = configs / "site.example.yaml"
    output_root = tmp_path / "workflow output with spaces"
    site.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "s21_catalogue": str(catalogue),
                    "s21_mask": str(mask),
                    "output_root": str(output_root),
                }
            }
        )
    )
    work = tmp_path / "working directory"
    work.mkdir()
    return {
        "experiment": experiment,
        "site": site,
        "work": work,
        "output_root": output_root,
        "catalogue": catalogue,
        "observation_dir": output_root
        / "derived/observations/s21_ds4_log_dispersion",
        "cache": tmp_path / "cache",
    }


def snakemake(case, *targets, dry_run=False, config=True):
    command = [
        sys.executable,
        "-m",
        "snakemake",
        "--snakefile",
        str(SNAKEFILE),
        "--directory",
        str(case["work"]),
        "--cores",
        "1",
        "--nocolor",
    ]
    if dry_run:
        command += ["--dry-run", "--printshellcmds"]
    command += list(targets)
    if config:
        command += [
            "--config",
            f"experiment={case['experiment']}",
            f"site={case['site']}",
        ]
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XDG_CACHE_HOME": str(case["cache"]),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    return result.returncode, result.stdout


def test_requires_an_explicit_site(workflow_case):
    code, output = snakemake(workflow_case, dry_run=True, config=False)
    assert code != 0
    assert "requires --config site=" in output


def test_complete_dag_declares_all_stages_and_products(workflow_case):
    code, output = snakemake(workflow_case, dry_run=True)
    assert code == 0, output
    for rule in (
        "resolve_catwise_experiment",
        "validate_catwise_data",
        "generate_catwise_mask",
        "prepare_catwise_observation",
        "infer_catwise",
        "package_catwise_analysis",
    ):
        assert f"rule {rule}:" in output
    for filename in (
        "samples_rnd-1.csv",
        "nflow_checkpoint_r1.npz",
        "software-state.yaml",
        "s21_nle_gaussian_poisson.zip.sha256",
    ):
        assert filename in output
    assert "--repository-root" in output
    assert "JAX_PLATFORMS=cpu" in output
    assert "-m dipolesbi.pipelines.based_catwise" in output
    assert "reference_observation_native_smoothed.png" in output
    run = load_catwise_run(workflow_case["experiment"], workflow_case["site"])
    assert run["inference"]["multiround"]["plot_save_dir"] == str(
        workflow_case["output_root"] / "results/s21_nle_gaussian_poisson"
    )


def test_real_preparation_is_noop_then_scopes_seed_prior_and_resource_changes(
    workflow_case,
):
    case = workflow_case
    code, output = snakemake(
        case, "resolve_catwise_experiment", "prepare_catwise_observation"
    )
    assert code == 0, output
    directory = case["observation_dir"]
    outputs = [
        directory / filename
        for filename in (
            "validation-report.yaml",
            "native_mask.npy",
            "reference_observation_native.npz",
            "reference_observation.npz",
            "mask.png",
            "reference_observation_native.png",
        )
    ]
    for filename in ("mask.png", "reference_observation_native.png"):
        assert (
            (directory / filename)
            .read_bytes()
            .startswith(b'\x89PNG\r\n\x1a\n')
        )
    timestamps = {path: path.stat().st_mtime_ns for path in outputs}
    resolved = (
        case["output_root"]
        / "derived/experiments/s21_nle_gaussian_poisson/experiment.resolved.yaml"
    )
    assert yaml.safe_load(resolved.read_text()) == load_catwise_run(
        case["experiment"], case["site"]
    )
    with np.load(directory / "reference_observation.npz") as observation:
        assert observation["x0"].shape == observation["mask"].shape == (193,)
        assert observation["mask"][-1]
        assert np.isfinite(observation["x0"][-1])
    code, output = snakemake(
        case,
        "resolve_catwise_experiment",
        "prepare_catwise_observation",
        dry_run=True,
    )
    assert code == 0, output
    assert "Nothing to be done" in output
    assert timestamps == {path: path.stat().st_mtime_ns for path in outputs}
    edit(
        case["experiment"],
        lambda c: c["execution"].update(seed=23, threads=2, mem_mb=128000),
    )
    edit(
        case["experiment"].parent / "priors_poisson.yaml",
        lambda c: c["parameters"][2].update(high=7),
    )
    code, output = snakemake(case, dry_run=True)
    assert code == 0, output
    assert "rule infer_catwise:" in output
    assert "rule resolve_catwise_experiment:" in output
    for rule in (
        "validate_catwise_data",
        "generate_catwise_mask",
        "prepare_catwise_observation",
    ):
        assert f"rule {rule}:" not in output
    assert timestamps == {path: path.stat().st_mtime_ns for path in outputs}
    edit(
        case["experiment"].parent / "observation.yaml",
        lambda c: c.update(downscale_nside=2),
    )
    code, output = snakemake(case, "prepare_catwise_observation", dry_run=True)
    assert code == 0, output
    assert "rule prepare_catwise_observation:" in output


def test_checksum_failure_blocks_mask_and_observation_creation(workflow_case):
    case = workflow_case
    with case["catalogue"].open("ab") as stream:
        stream.write(b"changed")
    code, output = snakemake(case, "prepare_catwise_observation")
    assert code != 0
    assert "validate_catwise_data" in output
    log = case["observation_dir"] / "validation.log"
    assert "checksum mismatch for catalogue" in log.read_text()
    assert not (case["observation_dir"] / "native_mask.npy").exists()
    assert not (case["observation_dir"] / "reference_observation.npz").exists()


def test_glass_experiment_rebuilds_summary_and_tracks_glass_settings(
    workflow_case,
):
    case = workflow_case
    code, output = snakemake(case, 'prepare_catwise_observation')
    assert code == 0, output
    case['experiment'] = (
        case['experiment'].parent / 's21_nle_gaussian_glass_with_cls.yaml'
    )
    edit(
        case['experiment'],
        lambda c: c['execution'].update(jax_platforms='cpu'),
    )
    code, output = snakemake(case, dry_run=True)
    assert code == 0, output
    assert 'rule infer_catwise:' in output
    assert 's21_nle_gaussian_glass_with_cls.zip' in output
    assert 'glass_power_spectrum.png' in output
    assert 'glass_power_spectra.npz' in output
    assert 'glass.py' in output
    for rule in (
        'validate_catwise_data',
        'generate_catwise_mask',
        'prepare_catwise_observation',
    ):
        assert f'rule {rule}:' in output
    resolved = load_catwise_run(case['experiment'], case['site'])
    assert resolved['execution']['backend'] == 'numpy'
    assert len(resolved['priors']) == 5
    code, output = snakemake(case, 'prepare_catwise_observation')
    assert code == 0, output
    directory = (
        case['output_root']
        / 'derived/observations'
        / resolved['observation']['observation_id']
    )
    with np.load(directory / 'reference_observation.npz') as data:
        assert data['x0'].shape == data['mask'].shape == (320,)
    code, output = snakemake(case, 'prepare_catwise_observation', dry_run=True)
    assert code == 0, output
    assert 'Nothing to be done' in output
    edit(
        case['experiment'].parent / 'model_glass.yaml',
        lambda c: c.update(glass_lmax=64),
    )
    code, output = snakemake(case, dry_run=True)
    assert code == 0, output
    assert 'rule infer_catwise:' in output
    assert 'rule prepare_catwise_observation:' in output


def test_inherited_observation_parent_is_a_workflow_input(workflow_case):
    case = workflow_case
    directory = case['experiment'].parent
    observation = directory / 'observation.yaml'
    parent = directory / 'observation_base.yaml'
    observation.rename(parent)
    observation.write_text('extends: observation_base\n')
    code, output = snakemake(case, 'prepare_catwise_observation')
    assert code == 0, output
    code, output = snakemake(case, 'prepare_catwise_observation', dry_run=True)
    assert code == 0, output
    assert 'Nothing to be done' in output
    edit(parent, lambda c: c.update(downscale_nside=2))
    code, output = snakemake(case, 'prepare_catwise_observation', dry_run=True)
    assert code == 0, output
    assert str(parent) in output
    assert 'rule prepare_catwise_observation:' in output


def test_numpy_worker_count_uses_snakemake_allocation(workflow_case):
    case = workflow_case
    edit(
        case['experiment'],
        lambda c: c['execution'].update(backend='numpy', threads=32),
    )
    code, output = snakemake(case, dry_run=True)
    assert code == 0, output
    # The test launcher offers one core, so Snakemake scales a 32-core request down.
    assert (
        load_catwise_run(case['experiment'], case['site'])['execution'][
            'threads'
        ]
        == 32
    )
    assert '--n-workers 1' in output
    assert '--n-workers 32' not in output
