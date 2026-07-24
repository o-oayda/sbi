"""Load, validate, and derive values for one selected RACS experiment."""

import re
import shlex
from pathlib import Path

from snakemake.exceptions import WorkflowError
from snakemake.utils import validate
import yaml


EXPERIMENT_SCHEMA_PATH = "schemas/racs-experiment.schema.yaml"
OBSERVATION_SCHEMA_PATH = "schemas/racs-observation.schema.yaml"
INFERENCE_SCHEMA_PATH = "schemas/racs-inference.schema.yaml"
EXPERIMENT_CONFIG_DIR = Path("workflow/configs/experiments")


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


# Resolve and validate the experiment, observation, and inference layers.
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

FINAL_ROUND = EXPERIMENT_ARGS["n_rounds"] - 1
RESULT_DIR = f"results/{EXPERIMENT_ID}"
OBSERVATION_DIR = f"derived/observations/{OBSERVATION_ID}"
OBSERVATION_PATH = f"{OBSERVATION_DIR}/reference_observation.npz"

SAMPLES_PATH = f"{RESULT_DIR}/samples_rnd-{FINAL_ROUND}.csv"
CHECKPOINT_PATH = f"{RESULT_DIR}/nflow_checkpoint_r{FINAL_ROUND}.npz"
RESULT_OBSERVATION_PATH = f"{RESULT_DIR}/reference_observation.npz"
EXPERIMENT_SNAPSHOT_PATH = f"{RESULT_DIR}/experiment.yaml"
INFERENCE_SNAPSHOT_PATH = f"{RESULT_DIR}/inference.yaml"
FINAL_OUTPUTS = (
    SAMPLES_PATH,
    CHECKPOINT_PATH,
    RESULT_OBSERVATION_PATH,
    EXPERIMENT_SNAPSHOT_PATH,
    INFERENCE_SNAPSHOT_PATH,
)
