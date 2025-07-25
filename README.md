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

## Requirements

- Python ≥ 3.12
- Poetry
- CUDA-compatible GPU (optional but recommended)