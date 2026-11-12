"""Configuration and dependency scopes for the separate CatWISE workflow."""

import hashlib
from importlib.util import find_spec
import os
from pathlib import Path
import sys

import yaml
from snakemake.exceptions import WorkflowError

REPOSITORY_ROOT = Path(workflow.basedir).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT))
from dipolesbi.pipelines.catwise_config import (
    CatwiseConfigError,
    DEFAULT_EXPERIMENT,
    load_catwise_run,
)

if not config.get("site"):
    raise WorkflowError("CatWISE requires --config site=/path/to/site.yaml")
EXPERIMENT_PATH = str(
    Path(config.get("experiment", DEFAULT_EXPERIMENT)).expanduser().resolve()
)
SITE_PATH = str(Path(config["site"]).expanduser().resolve())
try:
    RUN = load_catwise_run(EXPERIMENT_PATH, SITE_PATH)
except CatwiseConfigError as error:
    raise WorkflowError(str(error)) from error

PYTHON = sys.executable
PYTHONPATH = os.pathsep.join(
    filter(None, (str(REPOSITORY_ROOT), os.environ.get("PYTHONPATH")))
)
EXECUTION = RUN["execution"]
OUTPUT_ROOT = Path(RUN["paths"]["output_root"])
OBSERVATION_DIR = str(
    OUTPUT_ROOT / "derived/observations" / RUN["observation"]["observation_id"]
)
RESULT_DIR = str(OUTPUT_ROOT / "results" / RUN["experiment_id"])
ARTIFACT_DIR = str(OUTPUT_ROOT / "artifacts" / RUN["experiment_id"])
RESOLVED_CONFIG = str(
    OUTPUT_ROOT
    / "derived/experiments"
    / RUN["experiment_id"]
    / "experiment.resolved.yaml"
)
VALIDATION_REPORT = str(Path(OBSERVATION_DIR) / "validation-report.yaml")
NATIVE_MASK = str(Path(OBSERVATION_DIR) / "native_mask.npy")
PREPARED_OBSERVATION = str(Path(OBSERVATION_DIR) / "reference_observation.npz")
NATIVE_OBSERVATION = str(
    Path(OBSERVATION_DIR) / "reference_observation_native.npz"
)
ARCHIVE = str(Path(ARTIFACT_DIR) / f"{RUN['experiment_id']}.zip")
CHECKSUM = ARCHIVE + ".sha256"
GLASS_DIAGNOSTIC_OUTPUTS = (
    [
        str(Path(RESULT_DIR) / filename)
        for filename in ("glass_power_spectrum.png", "glass_power_spectra.npz")
    ]
    if RUN["model"]["position_model"] == "glass"
    else []
)
LAST_ROUND = EXECUTION["n_rounds"] - 1
CATALOGUE = RUN["paths"]["s21_catalogue"]
INPUT_MASK = RUN["paths"]["s21_mask"]
SOURCE_CATALOGUE = (
    []
    if RUN["paths"]["source_catalogue"] is None
    else [RUN["paths"]["source_catalogue"]]
)
CONFIG_PATHS = RUN["provenance"]["config_paths"]
OBSERVATION_CONFIG = RUN["provenance"]["config_layers"]["observation"]
CATSIM_ROOT = Path(find_spec("catsim").origin).parent


def files(*names):
    return [str(REPOSITORY_ROOT / name) for name in names]


SOFTWARE_LOCK = str(REPOSITORY_ROOT / "uv.lock")
CONFIG_IMPLEMENTATION = files(
    "dipolesbi/pipelines/catwise_config.py",
    "dipolesbi/pipelines/experiment_config.py",
)
CONFIG_SCHEMAS = sorted(
    str(path)
    for path in (REPOSITORY_ROOT / "workflow/schemas").glob(
        "catwise-*.schema.yaml"
    )
)
VALIDATION_IMPLEMENTATION = (
    files(
        "dipolesbi/pipelines/prepare_catwise_observation.py",
        "dipolesbi/pipelines/validate_racs_data.py",
    )
    + CONFIG_IMPLEMENTATION
)
MASK_IMPLEMENTATION = VALIDATION_IMPLEMENTATION + [
    str(CATSIM_ROOT / "utils/healsphere.py")
]
PREPARATION_IMPLEMENTATION = MASK_IMPLEMENTATION + files(
    "dipolesbi/pipelines/catwise_components.py",
    "dipolesbi/pipelines/catwise_pseudo_cl.py",
)
INFERENCE_IMPLEMENTATION = (
    PREPARATION_IMPLEMENTATION
    + files(
        "dipolesbi/pipelines/based_catwise.py",
        "dipolesbi/pipelines/catwise_diagnostics.py",
        "dipolesbi/tools/plotting.py",
        "dipolesbi/tools/maps.py",
        "dipolesbi/pipelines/package_catwise_analysis.py",
        "dipolesbi/tools/configs.py",
        "dipolesbi/tools/multiround_inferer.py",
        "dipolesbi/tools/priors_np.py",
        "dipolesbi/tools/priors_jax.py",
        "dipolesbi/tools/np_rngkey.py",
        "dipolesbi/tools/utils.py",
        "dipolesbi/tools/neural_flows.py",
        "dipolesbi/tools/transforms.py",
        "dipolesbi/tools/dataloader.py",
        "dipolesbi/tools/inference.py",
        "dipolesbi/tools/jax_ns.py",
        "dipolesbi/tools/model_config_io.py",
        "dipolesbi/tools/transform_io.py",
        "dipolesbi/tools/summary_diagnostics.py",
    )
    + [
        str(CATSIM_ROOT / name)
        for name in ("configs.py", "simulator.py", "simulator_jax.py")
    ]
)
PACKAGING_IMPLEMENTATION = (
    files("dipolesbi/pipelines/package_catwise_analysis.py")
    + CONFIG_IMPLEMENTATION
)

# GLASS and user-supplied named function implementations affect simulations,
# but do not change the authoritative observed map.
if RUN["model"]["position_model"] == "glass":
    INFERENCE_IMPLEMENTATION.append(str(CATSIM_ROOT / "glass.py"))
    for field in ("glass_dndz", "glass_bias"):
        module_name = RUN["model"][field].split(":", 1)[0]
        try:
            callable_spec = find_spec(module_name)
        except (ImportError, ValueError) as error:
            raise WorkflowError(
                f"Cannot locate {field} module: {module_name}"
            ) from error
        if callable_spec is None or callable_spec.origin is None:
            raise WorkflowError(f"Cannot locate {field} source: {module_name}")
        INFERENCE_IMPLEMENTATION.append(callable_spec.origin)


def fingerprint(mapping, implementation):
    digest = hashlib.sha256(yaml.safe_dump(mapping, sort_keys=True).encode())
    for filename in sorted(set(implementation)):
        digest.update(filename.encode())
        digest.update(Path(filename).read_bytes())
    return digest.hexdigest()


# Preparation depends on observation science and observed input paths. Priors,
# training choices, seeds, cache/source lookup paths and resource requests do not
# define the observed vector and therefore do not enter these fingerprints.
OBSERVATION_IDENTITY = {
    "observation": RUN["observation"],
    "paths": {key: RUN["paths"][key] for key in ("s21_catalogue", "s21_mask")},
}
VALIDATION_FINGERPRINT = fingerprint(
    OBSERVATION_IDENTITY, VALIDATION_IMPLEMENTATION + [SOFTWARE_LOCK]
)
MASK_FINGERPRINT = fingerprint(
    OBSERVATION_IDENTITY, MASK_IMPLEMENTATION + [SOFTWARE_LOCK]
)
PREPARATION_FINGERPRINT = fingerprint(
    OBSERVATION_IDENTITY, PREPARATION_IMPLEMENTATION + [SOFTWARE_LOCK]
)
INFERENCE_FINGERPRINT = fingerprint(
    RUN, INFERENCE_IMPLEMENTATION + [SOFTWARE_LOCK]
)
PACKAGING_FINGERPRINT = fingerprint(
    RUN, PACKAGING_IMPLEMENTATION + [SOFTWARE_LOCK]
)
