# S21 CatWISE NLE workflow

This workflow covers the S21 NLE experiment with Gaussian magnitude errors,
shared extra error and Poisson clustering. NPE and S22 are outside its scope.
The existing RACS workflow remains available through its own Snakefile.

The experiment YAML selects four scientific layers: `observation.yaml`,
`model.yaml`, `priors.yaml` and `inference.yaml`. It also declares the execution
budget, seed, batching, JAX platform and resources. The representative run
requests 50,000 simulations over 15 rounds. Floor division allocates 3,333 per
round, giving **49,995 effective simulations**. `no_ui: true` supports unattended
execution. Resolving configuration imports neither catsim nor JAX and does not
read catalogue data.

Create a machine-specific site file from the repository root:

```bash
mkdir -p workflow/configs/sites
cp workflow/configs/catwise/site.example.yaml workflow/configs/sites/catwise.local.yaml
```

Edit that file to point to the authoritative S21 observed catalogue, S21 mask
and an output root. `workflow/configs/sites/*.yaml` is ignored except for the
tracked site example. Relative site paths resolve against the site file's
directory; scientific layer paths resolve against the experiment file's
directory. Keep scientific choices in the scientific YAML layers.

`s21_catalogue` is the authoritative preselected observed catalogue, distinct
from `source_catalogue`, which optionally selects catsim's simulation source
catalogue. The workflow checks the observed catalogue and mask SHA-256 values
against `observation.yaml` before preparation or simulator construction. The
observed catalogue is cut with `w1_min < w1 < w1_max`; it receives no additional
colour cut. `w12_min` records the selection used for this preselected product
and is passed to the simulator. The native S21 RING mask is converted to a
Galactic NEST retain mask. **The north ecliptic mask is enabled by default**
(`mask_north_ecliptic: true`, agreeing with `use_noecl_mask: false`).

`source_catalogue: null` permits reuse of an existing compatible hashgrid;
if rebuilding is needed, provide the source catalogue path.
`hashgrid_cache: null` uses catsim's default user cache. Supplying paths selects
the existing catsim options; this workflow does not redesign its cache
generation, identity or reuse.

Inspect the complete resolved configuration and the planned workflow:

```bash
python -m dipolesbi.pipelines.catwise_config \
  --experiment workflow/configs/catwise/s21_nle_gaussian_poisson.yaml \
  --site workflow/configs/sites/catwise.local.yaml
snakemake --snakefile workflow/Snakefile.catwise --cores 1 --dry-run \
  --config experiment=workflow/configs/catwise/s21_nle_gaussian_poisson.yaml \
           site=workflow/configs/sites/catwise.local.yaml
```

The dry run does not launch inference. Configuration inspection needs no source
data; validation and preparation need the declared local S21 products. All
tests, scripts and interactive runs in the Codex CLI must use
`JAX_PLATFORMS=cpu` because that environment has no accessible GPU.

`execution.backend: jax` selects the JAX simulator. Independently,
`execution.jax_platforms: cuda` selects GPU execution for the production workflow.
Snakemake passes this value to each command, overriding the inherited
`JAX_PLATFORMS` environment variable. For a CPU workflow, set `jax_platforms: cpu`
in the experiment config; test fixtures do this explicitly. The configuration
inspection command above does not initialise JAX.

The workflow places prepared observations under
`<output_root>/derived/observations/<observation_id>/`, resolved configuration
under `<output_root>/derived/experiments/<experiment_id>/`, inference results
under `<output_root>/results/<experiment_id>/`, and archives under
`<output_root>/artifacts/<experiment_id>/`. It passes the archive output directory
explicitly to the packager.

The preparation stage writes `native_mask.npy`,
`reference_observation_native.npz` and `reference_observation.npz`.
Native-resolution HEALPix diagnostics are also saved: `mask.png` shows
the final retain mask, and `reference_observation_native.png` shows the masked
source counts. Both use Galactic NEST maps, matching the RACS projections.
The PNGs are declared workflow outputs and are regenerated if missing. Both NPZ
observations preserve `x0`, `mask` and `metadata_json`. The processed vector is
the downscaled NEST count map followed by native `log_dispersion`. The runner
preserves the prepared input as `prepared_observation.npz`, records input
identities and simulator-mask agreement in `validation-report.yaml`, and writes
the resolved experiment and invocation alongside inference outputs.
`software-state.yaml` records the Git commit, dirty status and environment-file
hashes at inference startup. Standalone inference should run from the source
checkout, or use `--repository-root /path/to/checkout`.

After inference, the standalone packaging command accepts the actual inference
result directory and preparation directory. Run it from the source checkout,
or supply `--repository-root /path/to/checkout` for the recorded Git provenance
and environment files:

```bash
python -m dipolesbi.pipelines.package_catwise_analysis \
  --experiment workflow/configs/catwise/s21_nle_gaussian_poisson.yaml \
  --site workflow/configs/sites/catwise.local.yaml \
  --result-dir /path/to/completed/inference-result \
  --observation-dir /path/to/prepared-observation
```

The archive defaults to
`artifacts/s21_nle_gaussian_poisson/s21_nle_gaussian_poisson.zip`, with a
`.zip.sha256` sidecar; `--output-dir` overrides the containing directory.
It includes all input YAML layers and the resolved config, authoritative input
identities and validation report, native and processed observations, native
mask, final-round samples and flow checkpoint, per-round evidence, model
configuration, invocation, `pyproject.toml`, `uv.lock`, and the inference-time
Git commit/dirty status. Packaging-time Git state is recorded separately. Each
payload has a SHA-256 checksum in `artifact-manifest.yaml`.
Source catalogues, hashgrid caches and multiround simulation datasets are
excluded; their configured paths remain recorded. Dirty Git state is explicitly
marked as not reproducible from the commit alone.

Packaging rejects missing products, changed resolved configuration, mismatched
input identities, and preparation or environment files changed after inference.
An explicit inference output-directory override remains valid. Verify archive
contents independently with:

```bash
python -c 'from dipolesbi.pipelines.package_catwise_analysis import verify_catwise_archive; verify_catwise_archive("artifacts/s21_nle_gaussian_poisson/s21_nle_gaussian_poisson.zip")'
```

Check the archive as a whole with `sha256sum -c` on its sidecar from the archive
directory. Member verification checks the internal inventory and checksums;
the external sidecar provides the checksum of the whole ZIP.

## Configuration inheritance

All CatWISE config layers, including experiment and site files, support the same
`extends` convention as RACS. The parent is a sibling YAML basename without the
`.yaml` suffix. For example, a new seed variant can contain only:

```yaml
extends: s21_nle_base
experiment_id: s21_nle_gaussian_poisson_seed1
execution:
  seed: 1
```

The parent supplies the common observation, inference config, budget, seed,
batching, resources, and remaining execution settings. Model, observation,
inference and prior layers can likewise extend a sibling base file.

Mappings merge recursively; child values override parent values. Lists and
scalars replace their parent values. This includes null: it is a value, not a
request to delete a mapping key. Multiple levels are supported; missing parents,
invalid parent names and cycles are rejected. Paths stay relative to the selected
config's directory, as before; parents are siblings in that same directory.
Schema and scientific validation run on the complete merged values.

Priors may be written as ordered lists (the existing format) or mappings keyed
by their short names. The named mapping form supports overriding one bound or
adding a parameter without repeating the whole prior list:

```yaml
extends: priors_base
parameters:
  lambda_clus:
    simulator_kwarg: lambda_clus
    low: 0
    high: 8
    dist_type: uniform
reference_theta:
  lambda_clus: 0.0
```

`priors_base.yaml` holds N, etaWX, speed and dipole direction, with their reference
values. `priors.yaml` adds Poisson clustering. Other supported variants can
reuse that common base. Existing parameter order is preserved and newly introduced
names append to it. The resolved output always uses an ordered parameter list,
so simulator and inference interfaces do not change. A child supplying a list
replaces the full inherited parameter mapping/list.

The inspection command emits all effective settings without `extends` keys.
`provenance.config_layers` records each layer's parent-first source paths;
every source is hashed, tracked as a workflow dependency, and included in the
analysis archive. Observation rules track only the observation's inheritance
tree, so model/prior/experiment changes do not invalidate the observed map.
