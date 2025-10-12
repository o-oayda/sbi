# dipolesbi

Fit dipoles using simulation-based inference.

## Quick Start

### Setup

Choose the appropriate setup based on your GPU:

**For RTX 5070 Ti (or other latest GPUs requiring CUDA 12.8):**
```bash
./setup_pytorch.sh --nightly
```

**For other GPUs (stable PyTorch):**
```bash
./setup_pytorch.sh
```

### Manual Setup

Alternatively, you can set up manually:
```bash
poetry install
# For RTX 5070 Ti, see PYTORCH_SETUP.md for additional PyTorch nightly installation
```

## Documentation

- [PyTorch CUDA 12.8 Setup](PYTORCH_SETUP.md) - Detailed GPU setup instructions
- Project structure and usage documentation coming soon

## Inspecting posterior results

After a multi-round inference run each output directory (for example
`fiducial_50k/20251011_155810_SEED0_NLE`) contains the weighted posterior
samples `samples_rnd-*.csv`.  You can visualise and compare these results with
the CLI helper:

```bash
python -m dipolesbi.tools.posterior_cli <run_dir> [<run_dir> ...] [options]
```

Key options:

- `--corner <path>` – write a GetDist corner plot for the runs. If several
  directories are provided the tool automatically keeps only the parameter
  columns present in every run, labels each corner with the directory name (or
  the name supplied via `--legend`), and overlays them in the same figure.
- `--sky-prob <path>` – draw the posterior sky probability (dipole direction)
  using Healpix. The first run is rendered in `cornflowerblue` with the filled
  map; additional runs are overlaid as contour outlines (default 1σ and 2σ for
  a 2‑D Gaussian) using the colour sequence
  `[cornflowerblue, tomato, #2ca02c, #d62728, ...]`.
- `--sky-smooth <sigma>` – set the spherical smoothing width (radians) before
  the map is projected (default `0.05`).
- `--legend <name ...>` – custom legend labels (must match the number of run
  directories). Otherwise, the directory names are used.
- `--round <n>` – inspect a specific inference round (defaults to the latest).
- `--logz-average-start <n>` – for each run, compute per-round log evidence for
  rounds `n` and above and report the inverse-variance weighted average (falling
  back to an unweighted average if no uncertainties are available).

Example: compare two runs in a single corner and sky plot while renaming the
legend entries and using a slightly wider sky smoothing kernel:

```bash
python -m dipolesbi.tools.posterior_cli \
  fiducial_50k/20251011_155810_SEED0_NLE \
  test/20251009_142336_SEED0_NLE \
  --legend fiducial baseline \
  --corner compare_corner.pdf \
  --sky-prob compare_sky.pdf \
  --sky-smooth 0.08
```

The CLI also prints per-run summary statistics and log evidence values so you
can quickly gauge differences between runs. Posterior weights from each CSV
are honoured automatically in both the corner plots and sky projections.

## Requirements

- Python ≥ 3.12
- Poetry
- CUDA-compatible GPU (optional but recommended)
