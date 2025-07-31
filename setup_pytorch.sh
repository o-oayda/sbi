#!/bin/bash

# Setup script for dipolesbi project
# Supports both stable and nightly PyTorch installations

set -e  # Exit on any error

# Default values
NIGHTLY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nightly)
            NIGHTLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Setup script for dipolesbi project with PyTorch support"
            echo ""
            echo "Options:"
            echo "  --nightly    Install PyTorch nightly with CUDA 12.8 support (for RTX 5070 Ti)"
            echo "  -h, --help   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                # Install stable PyTorch (default)"
            echo "  $0 --nightly      # Install PyTorch nightly with CUDA 12.8"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Setting up dipolesbi project..."

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Please install Poetry first:"
    echo "   curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Install dependencies with Poetry
echo "📦 Installing dependencies with Poetry..."
poetry install

if [ "$NIGHTLY" = true ]; then
    echo "🔥 Installing PyTorch nightly (dev20250724) with CUDA 12.8 support..."
    poetry run pip install --upgrade --index-url https://download.pytorch.org/whl/nightly/cu128 torch==2.9.0.dev20250724+cu128 torchvision==0.24.0.dev20250724+cu128 torchaudio==2.8.0.dev20250724+cu128
    
    echo "🧪 Testing GPU detection..."
    poetry run python -c "import torch; print('✅ PyTorch version:', torch.__version__); print('✅ CUDA available:', torch.cuda.is_available()); print('✅ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
    
    echo "🎉 Setup complete! Your RTX 5070 Ti is ready for PyTorch nightly."
else
    echo "🧪 Testing PyTorch installation..."
    poetry run python -c "import torch; print('✅ PyTorch version:', torch.__version__); print('✅ CUDA available:', torch.cuda.is_available()); print('✅ GPU count:', torch.cuda.device_count())"
    
    echo "✅ Setup complete! Stable PyTorch is installed."
    echo "💡 Tip: For RTX 5070 Ti support, run: $0 --nightly"
fi
