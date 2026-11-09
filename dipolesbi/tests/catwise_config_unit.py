"""Contract tests for the layered, configuration-only S21 NLE loader."""

from dataclasses import fields
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from dipolesbi.pipelines.catwise_config import (
    CatwiseConfigError,
    DEFAULT_EXPERIMENT,
    load_catwise_run,
    main,
)


@pytest.fixture
def configs(tmp_path):
    directory = tmp_path / "configs"
    shutil.copytree(DEFAULT_EXPERIMENT.parent, directory)
    site = directory / "site.example.yaml"
    site.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "s21_catalogue": "../inputs/s21.fits",
                    "s21_mask": "../inputs/mask.npy",
                    "output_root": "../results",
                }
            }
        )
    )
    return directory / DEFAULT_EXPERIMENT.name, site


def edit(path, update):
    value = yaml.safe_load(path.read_text())
    update(value)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def test_representative_contract_resolves_without_data_files(configs):
    experiment, site = configs
    run = load_catwise_run(experiment, site)
    assert run["observation"]["catalogue_version"] == "S21"
    assert run["observation"]["native_nside"] == 64
    assert run["observation"]["downscale_nside"] == 4
    assert run["observation"]["native_count_summary"] == "log_dispersion"
    assert run["model"]["max_cluster_children_per_parent"] == 6
    assert run["model"]["simulate_clustering"] == "poisson"
    assert run["execution"] == {
        "backend": "jax",
        "requested_simulations": 50_000,
        "n_rounds": 15,
        "seed": 0,
        "jax_batch_size": 10,
        "chunk_size": 25_000,
        "jax_platforms": "cuda",
        "threads": 1,
        "mem_mb": 64_000,
        "no_ui": True,
        "simulations_per_round": 3333,
        "effective_simulations": 49_995,
    }
    assert [p["short_name"] for p in run["priors"]] == [
        "N",
        "etaWX",
        "D",
        "phi",
        "theta",
        "lambda_clus",
    ]
    assert run["priors"][0]["low"] == pytest.approx(7.477121254719663)
    assert run["priors"][0]["high"] == pytest.approx(7.6020599913279625)
    assert run["fixed_parameters"] == {}
    assert run["paths"]["source_catalogue"] is None
    assert run["paths"]["hashgrid_cache"] is None
    assert run["paths"]["s21_catalogue"] == str(
        site.parent.parent / "inputs/s21.fits"
    )
    assert not Path(run["paths"]["s21_catalogue"]).exists()
    multiround = run["inference"]["multiround"]
    assert multiround["simulation_budget"] == 50_000
    assert multiround["effective_simulations"] == 49_995
    assert multiround["map_ndim"] == 192
    assert multiround["summary_ndim"] == 1
    assert multiround["reference_theta"] == run["reference_theta"]
    assert multiround["plot_save_dir"] == str(
        site.parent.parent / "results/results" / run["experiment_id"]
    )
    for path, digest in run["provenance"]["config_sha256"].items():
        assert digest == hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_all_inference_dataclass_inputs_are_materialized(configs):
    from dipolesbi.tools.configs import (
        DataTransformSpec,
        MultiRoundInfererConfig,
        NeuralFlowConfig,
        ThetaTransformSpec,
        TrainingConfig,
    )

    run = load_catwise_run(*configs)
    inference = run["inference"]
    for section, cls, supplied_elsewhere in (
        ("training", TrainingConfig, set()),
        ("flow", NeuralFlowConfig, {"mode"}),
        ("data_transform", DataTransformSpec, set()),
        ("theta_transform", ThetaTransformSpec, set()),
    ):
        assert (
            set(inference[section])
            == {f.name for f in fields(cls) if f.init} - supplied_elsewhere
        )
    expected = {f.name for f in fields(MultiRoundInfererConfig)} | {
        "effective_simulations"
    }
    assert set(inference["multiround"]) == expected
    assert inference["flow"]["architecture"] == 4 * ["MAF"] + [
        "surjective_MAF"
    ] + 6 * ["MAF"]
    assert inference["training"]["learning_rate"] == 1e-4
    assert inference["training"]["min_lr_ratio"] == 1.0
    assert inference["flow"]["decoder_n_layers"] == 4
    assert inference["flow"]["decoder_n_neurons"] == 128
    assert inference["flow"]["conditioner_n_layers"] == 4
    assert inference["flow"]["conditioner_n_neurons"] == 256
    assert inference["data_transform"]["zscore_method"] == "batchwise"


def test_configurable_ecliptic_mask_requires_matching_model_flag(configs):
    experiment, site = configs
    edit(
        experiment.parent / "observation.yaml",
        lambda c: c.update(mask_north_ecliptic=False),
    )
    with pytest.raises(CatwiseConfigError, match="disagrees"):
        load_catwise_run(experiment, site)
    edit(
        experiment.parent / "model.yaml",
        lambda c: c.update(use_noecl_mask=True),
    )
    run = load_catwise_run(experiment, site)
    assert run["observation"]["mask_north_ecliptic"] is False
    assert run["model"]["use_noecl_mask"] is True


@pytest.mark.parametrize(
    "filename,field,value",
    [
        (DEFAULT_EXPERIMENT.name, "typo", True),
        ("observation.yaml", "catalogue_version", "S22"),
        ("observation.yaml", "downscale_nside", 3),
        ("observation.yaml", "catalogue_sha256", "bad"),
        ("observation.yaml", "w1_min", 8.9),
        ("observation.yaml", "w1_max", 16.6),
        ("observation.yaml", "w12_min", 0.9),
        ("model.yaml", "magnitude_error_dist", "students-t"),
        ("model.yaml", "use_common_extra_error", False),
        ("model.yaml", "simulate_clustering", "other"),
        ("inference.yaml", "mode", "NPE"),
        ("priors.yaml", "fixed_parameters", {"observer_speed": 2}),
    ],
)
def test_rejects_unsupported_or_unknown_science(
    configs, filename, field, value
):
    experiment, site = configs
    edit(
        experiment.parent / filename,
        lambda config: config.update({field: value}),
    )
    with pytest.raises(CatwiseConfigError):
        load_catwise_run(experiment, site)


@pytest.mark.parametrize(
    "update,match",
    [
        (lambda c: c["execution"].update(n_rounds=50_001), "at least one"),
        (lambda c: c["execution"].update(jax_batch_size=0), "jax_batch_size"),
        (lambda c: c["execution"].update(chunk_size=0), "chunk_size"),
        (lambda c: c["execution"].update(seed=-1), "seed"),
    ],
)
def test_rejects_invalid_execution(configs, update, match):
    experiment, site = configs
    edit(experiment, update)
    with pytest.raises(CatwiseConfigError, match=match):
        load_catwise_run(experiment, site)


def test_semantic_prior_and_observation_errors(configs):
    experiment, site = configs
    edit(
        experiment.parent / "priors.yaml",
        lambda c: c["parameters"][1].update(low=9),
    )
    with pytest.raises(CatwiseConfigError, match="low < high"):
        load_catwise_run(experiment, site)
    edit(
        experiment.parent / "priors.yaml",
        lambda c: c["parameters"][1].update(low=0),
    )
    edit(experiment.parent / "observation.yaml", lambda c: c.update(w1_min=17))
    with pytest.raises(CatwiseConfigError, match="w1_max"):
        load_catwise_run(experiment, site)


def test_missing_files_and_yaml_fail_with_configuration_error(configs):
    experiment, site = configs
    edit(experiment, lambda c: c.update(model="missing.yaml"))
    with pytest.raises(CatwiseConfigError, match="Cannot read configuration"):
        load_catwise_run(experiment, site)
    experiment.write_text("[not, a, mapping]")
    with pytest.raises(CatwiseConfigError, match="YAML mapping"):
        load_catwise_run(experiment, site)


@pytest.mark.parametrize(
    "update,match",
    [
        (
            lambda c: c["reference_theta"].update(observer_speed=float("nan")),
            "finite",
        ),
        (
            lambda c: c["reference_theta"].update(observer_speed=20),
            "outside prior",
        ),
        (lambda c: c["parameters"].reverse(), "must follow"),
    ],
)
def test_prior_values_and_order_are_validated(configs, update, match):
    experiment, site = configs
    edit(experiment.parent / "priors.yaml", update)
    with pytest.raises(CatwiseConfigError, match=match):
        load_catwise_run(experiment, site)


def test_rejects_a_flow_flag_that_scenario_would_silently_overwrite(configs):
    experiment, site = configs
    edit(
        experiment.parent / "inference.yaml",
        lambda c: c["flow"].update(embed_target_transform_in_flow=True),
    )
    with pytest.raises(CatwiseConfigError, match="must match"):
        load_catwise_run(experiment, site)


def test_cli_outputs_reviewable_resolved_yaml_without_heavy_imports(
    configs, tmp_path
):
    output = tmp_path / "resolved.yaml"
    assert (
        main(
            [
                "--experiment",
                str(configs[0]),
                "--site",
                str(configs[1]),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (
        yaml.safe_load(output.read_text())["execution"][
            "effective_simulations"
        ]
        == 49_995
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from dipolesbi.pipelines.catwise_config import load_catwise_run; "
                f"load_catwise_run({str(configs[0])!r}, {str(configs[1])!r}); "
                "assert not any(k in sys.modules for k in ('snakemake', 'jax', 'catsim'))"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
