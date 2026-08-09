"""Load, validate, and derive values for one selected RACS experiment."""

import hashlib
import math
import re
import shlex
from pathlib import Path

from snakemake.exceptions import WorkflowError
from snakemake.utils import validate
import yaml

from dipolesbi.pipelines.package_racs_analysis import implementation_fingerprint


EXPERIMENT_SCHEMA_PATH = "schemas/racs-experiment.schema.yaml"
OBSERVATION_SCHEMA_PATH = "schemas/racs-observation.schema.yaml"
INFERENCE_SCHEMA_PATH = "schemas/racs-inference.schema.yaml"
DATASET_SCHEMA_PATH = "schemas/racs-datasets.schema.yaml"
SITE_SCHEMA_PATH = "schemas/racs-site.schema.yaml"
FILE_COLLECTION_SCHEMA_PATH = "schemas/racs-file-collection.schema.yaml"
EXPERIMENT_CONFIG_DIR = Path("workflow/configs/experiments")
DATASET_REGISTRY_PATH = Path("workflow/configs/datasets.yaml")
PREPARATION_IMPLEMENTATION_PATHS = (
    Path("dipolesbi/pipelines/prepare_racs_observation.py"),
    Path("dipolesbi/pipelines/racs_observation_helpers.py"),
    Path("dipolesbi/pipelines/summary_stats.py"),
    Path("uv.lock"),
)
INFERENCE_IMPLEMENTATION_PATHS = (
    Path("dipolesbi/pipelines/based_racs.py"),
    Path("dipolesbi/pipelines/racs_observation_helpers.py"),
    Path("dipolesbi/pipelines/summary_stats.py"),
    Path("dipolesbi/tools"),
    Path("uv.lock"),
)


def content_fingerprint(paths):
    """Fingerprint live contents of explicitly scoped files and Python trees."""
    files = set()
    for path in map(Path, paths):
        if path.is_dir():
            files.update(path.rglob("*.py"))
        elif path.is_file():
            files.add(path)
        else:
            raise WorkflowError(f"Fingerprint input does not exist: {path}")

    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_and_validate_yaml(path, schema_path, description):
    """Load a YAML mapping and validate it against a JSON Schema."""
    path = Path(path)
    if not path.is_file():
        raise WorkflowError(f"{description} config does not exist: {path}")

    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise WorkflowError(
            f"{description} config must contain a YAML mapping: {path}"
        )

    validate(values, schema_path)
    return values


def selected_experiment_config_path(workflow_config):
    """Resolve the experiment selected with ``--config experiment=<name>``."""
    experiment_name = workflow_config.get("experiment")
    if not isinstance(experiment_name, str) or not experiment_name:
        raise WorkflowError(
            "Select an experiment with '--config experiment=<experiment-name>'."
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_name):
        raise WorkflowError(f"Invalid experiment name: {experiment_name}")

    return experiment_name, EXPERIMENT_CONFIG_DIR / f"{experiment_name}.yaml"


def merge_distinct_arguments(observation_args, experiment_args):
    """Merge argument mappings while rejecting ambiguous ownership."""
    duplicated_args = set(experiment_args) & set(observation_args)
    if duplicated_args:
        duplicated = ", ".join(sorted(duplicated_args))
        raise WorkflowError(
            "Observation-owned arguments must only be declared in the observation "
            f"config; duplicated argument(s): {duplicated}"
        )
    return {**observation_args, **experiment_args}


def command_line_arguments(arguments):
    """Translate a YAML argument mapping into shell-safe argparse tokens."""
    tokens = []

    for name, value in arguments.items():
        flag = f"--{name}"

        if isinstance(value, bool):
            if value:
                tokens.append(flag)
        elif value is None:
            continue
        elif isinstance(value, list):
            tokens.append(flag)
            tokens.extend(str(item) for item in value)
        else:
            tokens.extend([flag, str(value)])

    return " ".join(shlex.quote(token) for token in tokens)


def dataset_definition(dataset_id, expected_type, dataset_registry):
    """Return one registry entry needed to construct the DAG."""
    dataset = dataset_registry["datasets"].get(dataset_id)
    if dataset is None:
        raise WorkflowError(f"Dataset is not declared in the registry: {dataset_id}")
    if dataset["type"] != expected_type:
        raise WorkflowError(
            f"Dataset '{dataset_id}' has type '{dataset['type']}', expected "
            f"'{expected_type}'."
        )
    return dataset


def resolve_site_path(dataset_id, site_config):
    """Resolve the machine-specific path for one logical dataset."""
    location = site_config["data_locations"].get(dataset_id)
    if location is None:
        raise WorkflowError(
            f"Site config does not define a path for dataset: {dataset_id}"
        )

    path = Path(location["path"]).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise WorkflowError(
            f"Configured path for dataset '{dataset_id}' does not exist: {path}"
        ) from error
    return resolved


def resolve_file_dataset(dataset_id, dataset_registry, site_config):
    """Resolve one logical file dataset through the selected site mapping."""
    dataset_definition(dataset_id, "file", dataset_registry)
    resolved = resolve_site_path(dataset_id, site_config)
    if not resolved.is_file():
        raise WorkflowError(
            f"Configured path for dataset '{dataset_id}' is not a file: {resolved}"
        )
    return resolved


def resolve_file_collection_dataset(dataset_id, dataset_registry, site_config):
    """Resolve the root, declared files, and matching files for the DAG."""
    dataset = dataset_definition(dataset_id, "file_collection", dataset_registry)
    resolved_root = resolve_site_path(dataset_id, site_config)
    if not resolved_root.is_dir():
        raise WorkflowError(
            f"Configured path for dataset '{dataset_id}' is not a directory: "
            f"{resolved_root}"
        )

    manifest_path = Path(dataset["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = DATASET_REGISTRY_PATH.parent / manifest_path
    manifest_path = manifest_path.resolve(strict=True)
    manifest = load_and_validate_yaml(
        manifest_path,
        FILE_COLLECTION_SCHEMA_PATH,
        f"Manifest for dataset '{dataset_id}'",
    )

    resolved_files = []
    for entry in manifest["files"]:
        candidate = resolved_root / entry["relative_path"]
        try:
            resolved_file = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            raise WorkflowError(
                f"Dataset '{dataset_id}' is missing manifest file: {candidate}"
            ) from error
        if not resolved_file.is_file():
            raise WorkflowError(
                f"Dataset '{dataset_id}' manifest entry is not a file: {candidate}"
            )
        resolved_files.append(resolved_file)

    matched_files = tuple(
        sorted(
            path.resolve(strict=True)
            for path in resolved_root.glob(manifest["file_glob"])
            if path.is_file()
        )
    )
    validation_files = tuple(
        dict.fromkeys((*resolved_files, *matched_files))
    )
    return (
        resolved_root,
        tuple(resolved_files),
        validation_files,
        manifest_path,
    )


# Resolve and validate the experiment, observation, and inference layers.
DATASET_REGISTRY = load_and_validate_yaml(
    DATASET_REGISTRY_PATH,
    DATASET_SCHEMA_PATH,
    "Dataset registry",
)
if "data_locations" not in config:
    raise WorkflowError(
        "Select a machine-specific site config with "
        "'--configfile workflow/configs/sites/<site>.yaml'."
    )
SITE_CONFIG = {"data_locations": config["data_locations"]}
validate(SITE_CONFIG, SITE_SCHEMA_PATH)

EXPERIMENT_NAME, EXPERIMENT_CONFIG_PATH = selected_experiment_config_path(config)
EXPERIMENT = load_and_validate_yaml(
    EXPERIMENT_CONFIG_PATH,
    EXPERIMENT_SCHEMA_PATH,
    "Experiment",
)
if EXPERIMENT["experiment_id"] != EXPERIMENT_NAME:
    raise WorkflowError(
        f"Selected experiment '{EXPERIMENT_NAME}' declares experiment_id "
        f"'{EXPERIMENT['experiment_id']}'. The names must match."
    )

OBSERVATION_CONFIG_PATH = Path(EXPERIMENT["observation_config"])
OBSERVATION = load_and_validate_yaml(
    OBSERVATION_CONFIG_PATH,
    OBSERVATION_SCHEMA_PATH,
    "Observation",
)
if (
    OBSERVATION["args"]["temperature_fallback"] == "reference"
    and not math.isfinite(OBSERVATION["args"]["paf_reference_temp_c"])
):
    raise WorkflowError(
        "Reference temperature fallback requires a finite "
        "'paf_reference_temp_c'."
    )
if OBSERVATION["observation_id"] != OBSERVATION_CONFIG_PATH.stem:
    raise WorkflowError(
        f"Observation config '{OBSERVATION_CONFIG_PATH}' declares observation_id "
        f"'{OBSERVATION['observation_id']}'. The filename and ID must match."
    )
CATALOGUE_DATASET_ID = OBSERVATION["datasets"]["catalogue"]
CATALOGUE_PATH = resolve_file_dataset(
    CATALOGUE_DATASET_ID,
    DATASET_REGISTRY,
    SITE_CONFIG,
)
PAF_TEMPERATURE_DATASET_ID = OBSERVATION["datasets"].get("paf_temperatures")
if PAF_TEMPERATURE_DATASET_ID is None:
    PAF_TEMPERATURE_DIR = None
    PAF_TEMPERATURE_FILES = ()
    PAF_TEMPERATURE_VALIDATION_FILES = ()
    PAF_TEMPERATURE_MANIFEST_PATH = None
    PAF_VALIDATION_CLI_ARGS = ""
    PAF_PREPARATION_CLI_ARGS = ""
    PAF_INFERENCE_CLI_ARGS = ""
    PAF_PACKAGING_CLI_ARGS = ""
else:
    (
        PAF_TEMPERATURE_DIR,
        PAF_TEMPERATURE_FILES,
        PAF_TEMPERATURE_VALIDATION_FILES,
        PAF_TEMPERATURE_MANIFEST_PATH,
    ) = resolve_file_collection_dataset(
        PAF_TEMPERATURE_DATASET_ID,
        DATASET_REGISTRY,
        SITE_CONFIG,
    )
    PAF_VALIDATION_CLI_ARGS = " ".join(
        (
            "--paf-id",
            shlex.quote(PAF_TEMPERATURE_DATASET_ID),
            "--paf-root",
            shlex.quote(str(PAF_TEMPERATURE_DIR)),
            "--paf-manifest",
            shlex.quote(str(PAF_TEMPERATURE_MANIFEST_PATH)),
        )
    )
    PAF_PREPARATION_CLI_ARGS = (
        "--paf-temperature-data-dir "
        + shlex.quote(str(PAF_TEMPERATURE_DIR))
    )
    PAF_INFERENCE_CLI_ARGS = (
        "--paf_temperature_data_dir "
        + shlex.quote(str(PAF_TEMPERATURE_DIR))
    )
    PAF_PACKAGING_CLI_ARGS = (
        "--paf-manifest " + shlex.quote(str(PAF_TEMPERATURE_MANIFEST_PATH))
    )

NOISE_MAP_DATASET_ID = OBSERVATION["datasets"]["noise_maps"]
(
    NOISE_MAP_DIR,
    NOISE_MAP_FILES,
    NOISE_MAP_VALIDATION_FILES,
    NOISE_MAP_MANIFEST_PATH,
) = resolve_file_collection_dataset(
    NOISE_MAP_DATASET_ID,
    DATASET_REGISTRY,
    SITE_CONFIG,
)
NOISE_MAP_PREPARATION_CLI_ARGS = (
    "--noisemap-data-dir " + shlex.quote(str(NOISE_MAP_DIR))
)
NOISE_MAP_INFERENCE_CLI_ARGS = (
    "--noisemap_data_dir " + shlex.quote(str(NOISE_MAP_DIR))
)
NOISE_MAP_VALIDATION_CLI_ARGS = " ".join(
    (
        "--noise-map-id",
        shlex.quote(NOISE_MAP_DATASET_ID),
        "--noise-map-root",
        shlex.quote(str(NOISE_MAP_DIR)),
        "--noise-map-manifest",
        shlex.quote(str(NOISE_MAP_MANIFEST_PATH)),
    )
)

INFERENCE_CONFIG_PATH = Path(EXPERIMENT["inference_config"])
INFERENCE = load_and_validate_yaml(
    INFERENCE_CONFIG_PATH,
    INFERENCE_SCHEMA_PATH,
    "Inference",
)
if INFERENCE["inference_id"] != INFERENCE_CONFIG_PATH.stem:
    raise WorkflowError(
        f"Inference config '{INFERENCE_CONFIG_PATH}' declares inference_id "
        f"'{INFERENCE['inference_id']}'. The filename and ID must match."
    )


# Expose concise, derived values for the rules in the main Snakefile.
EXPERIMENT_ID = EXPERIMENT["experiment_id"]
OBSERVATION_ID = OBSERVATION["observation_id"]
EXECUTION = EXPERIMENT["execution"]
EXPERIMENT_ARGS = EXPERIMENT["args"]
OBSERVATION_ARGS = OBSERVATION["args"]
INFERENCE_ARGS = merge_distinct_arguments(OBSERVATION_ARGS, EXPERIMENT_ARGS)
IMPLEMENTATION_FINGERPRINT = implementation_fingerprint(Path.cwd())
PREPARATION_IMPLEMENTATION_FINGERPRINT = content_fingerprint(
    PREPARATION_IMPLEMENTATION_PATHS
)
INFERENCE_IMPLEMENTATION_FINGERPRINT = content_fingerprint(
    INFERENCE_IMPLEMENTATION_PATHS
)

FINAL_ROUND = EXPERIMENT_ARGS["n_rounds"] - 1
RESULT_DIR = f"results/{EXPERIMENT_ID}"
OBSERVATION_DIR = f"derived/observations/{OBSERVATION_ID}"
OBSERVATION_PATH = f"{OBSERVATION_DIR}/reference_observation.npz"
NATIVE_OBSERVATION_PATH = f"{OBSERVATION_DIR}/reference_observation_native.npz"
DATA_VALIDATION_DIR = (
    f"derived/data-validation/{CATALOGUE_DATASET_ID}--"
    f"{PAF_TEMPERATURE_DATASET_ID or 'open-meteo'}--{NOISE_MAP_DATASET_ID}"
)
DATA_VALIDATION_PATH = f"{DATA_VALIDATION_DIR}/validation-report.yaml"

SAMPLES_PATH = f"{RESULT_DIR}/samples_rnd-{FINAL_ROUND}.csv"
ROUND_EVIDENCE_PATH = f"{RESULT_DIR}/epoch_lnZ.npy"
CHECKPOINT_PATH = f"{RESULT_DIR}/nflow_checkpoint_r{FINAL_ROUND}.npz"
RESULT_OBSERVATION_PATH = f"{RESULT_DIR}/reference_observation.npz"
MODEL_CONFIG_PATH = f"{RESULT_DIR}/model_config.json"
CONFIGS_PATH = f"{RESULT_DIR}/configs.txt"
RUN_COMMAND_PATH = f"{RESULT_DIR}/run_command.txt"
ARTIFACT_DIR = f"artifacts/{EXPERIMENT_ID}"
ANALYSIS_ARCHIVE_PATH = f"{ARTIFACT_DIR}/{EXPERIMENT_ID}.analysis.zip"
ANALYSIS_CHECKSUM_PATH = f"{ANALYSIS_ARCHIVE_PATH}.sha256"
FINAL_OUTPUTS = (
    ANALYSIS_ARCHIVE_PATH,
    ANALYSIS_CHECKSUM_PATH,
)
