# PyTorch CUDA 12.8 Setup for RTX 5070 Ti

This project uses PyTorch nightly builds to support the RTX 5070 Ti GPU with CUDA 12.8.

## Installation

### Option 1: Automated Setup (Recommended)

**For stable PyTorch (works with most GPUs):**
```bash
./setup_pytorch.sh
```

**For nightly PyTorch with CUDA 12.8 (required for RTX 5070 Ti):**
```bash
./setup_pytorch.sh --nightly
```

### Option 2: Manual Setup

1. Install dependencies with Poetry:
```bash
poetry install
```

2. (Optional) Install PyTorch nightly with CUDA 12.8 support:
```bash
poetry run pip install --upgrade --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio
```

## Verification

Test that your GPU is properly detected:

```bash
poetry run python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## Current Setup

- PyTorch: 2.9.0.dev20250724+cu128
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 5070 Ti

## Understanding Version Tracking

You may notice that `poetry show torch` displays version 2.5.1, while Python actually uses 2.9.0.dev20250724+cu128. This is expected and correct:

- **Poetry tracking**: Shows the stable version (2.5.1) from the lock file
- **Python runtime**: Uses the nightly version (2.9.0.dev20250724+cu128) installed by pip
- **Why this works**: When multiple versions are present, Python uses the more recent installation

This hybrid approach gives us:
- ✅ Clean dependency management via Poetry for all other packages
- ✅ Latest PyTorch nightly with RTX 5070 Ti support
- ✅ Reproducible builds (since the nightly version is pinned)

## Note

The dependency resolver in Poetry doesn't handle PyTorch nightly builds well, so we use pip within the Poetry environment for PyTorch installation while letting Poetry handle all other dependencies. This is a common and recommended pattern for projects requiring cutting-edge GPU support.
