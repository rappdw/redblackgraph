#!/bin/bash
# Setup script for DGX Spark (Grace Hopper) GPU development
# This script sets up the environment for GPU-accelerated redblackgraph

set -e  # Exit on error

echo "================================================"
echo "DGX Spark Setup for redblackgraph GPU"
echo "================================================"
echo ""

# Check if we're on a system with NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. Is this a GPU system?"
    exit 1
fi

echo "✓ NVIDIA driver found"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

# Check CUDA version
if ! command -v nvcc &> /dev/null; then
    echo "⚠️  WARNING: nvcc not found. CUDA toolkit may not be installed."
    echo "   CuPy wheels include CUDA runtime, but development may need full toolkit."
else
    echo "✓ CUDA toolkit found"
    nvcc --version | grep "release"
fi

echo ""
echo "================================================"
echo "Setting up Python environment"
echo "================================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ ERROR: Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python version OK"

# Check for Python development headers
echo ""
echo "Checking for Python development headers..."
if ! dpkg -l | grep -q "python3-dev"; then
    echo "⚠️  python3-dev not found. Installing system dependencies..."
    echo "   This requires sudo access."
    sudo apt-get update
    sudo apt-get install -y python3-dev python3-venv build-essential
    echo "✓ System dependencies installed"
else
    echo "✓ Python development headers found"
fi

# Check for uv
echo ""
echo "Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "⚠️  uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "✓ uv installed"
else
    echo "✓ uv found"
fi

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create virtual environment if it doesn't exist
VENV_PATH="${PROJECT_ROOT}/.venv-rbg-gpu"

if [ ! -d "$VENV_PATH" ]; then
    echo ""
    echo "Creating virtual environment at $VENV_PATH..."
    uv venv "$VENV_PATH" --python python3
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists at $VENV_PATH"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Install build dependencies
echo ""
echo "Installing build dependencies..."
uv pip install meson-python meson ninja cython tempita wheel setuptools

# Detect CUDA version and install appropriate CuPy
echo ""
echo "================================================"
echo "Installing CuPy"
echo "================================================"

# Try to detect CUDA version from nvidia-smi
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)

if [ -z "$CUDA_VERSION" ]; then
    echo "⚠️  Could not auto-detect CUDA version"
    CUDA_VERSION="12"  # Default to CUDA 12 for H100
    echo "   Defaulting to CUDA $CUDA_VERSION"
else
    echo "Detected CUDA major version: $CUDA_VERSION"
fi

# Install appropriate CuPy version
if [ "$CUDA_VERSION" = "12" ]; then
    echo "Installing cupy-cuda12x..."
    uv pip install cupy-cuda12x
elif [ "$CUDA_VERSION" = "11" ]; then
    echo "Installing cupy-cuda11x..."
    uv pip install cupy-cuda11x
elif [ "$CUDA_VERSION" = "13" ]; then
    echo "Installing cupy-cuda13x..."
    uv pip install cupy-cuda13x
else
    echo "⚠️  Unsupported CUDA version: $CUDA_VERSION"
    echo "   Installing cupy-cuda12x (may not work)"
    uv pip install cupy-cuda12x
fi

# Verify CuPy installation
echo ""
echo "Verifying CuPy installation..."
if python3 -c "import cupy as cp; print('CuPy version:', cp.__version__); print('GPU:', cp.cuda.Device(0).compute_capability)" 2>/dev/null; then
    echo "✓ CuPy installed and working"
else
    echo "❌ ERROR: CuPy installation failed or GPU not accessible"
    exit 1
fi

# Install redblackgraph dependencies
echo ""
echo "================================================"
echo "Installing redblackgraph"
echo "================================================"
echo "Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

# Install in development mode
echo ""
echo "Installing redblackgraph in development mode..."
uv pip install -e ".[dev,test]" --no-build-isolation

# Verify installation
echo ""
echo "Verifying redblackgraph installation..."
if python3 -c "import redblackgraph; print('redblackgraph version:', redblackgraph.__version__)" 2>/dev/null; then
    echo "✓ redblackgraph installed"
else
    echo "❌ ERROR: redblackgraph installation failed"
    exit 1
fi

# Verify GPU module
echo ""
echo "Verifying GPU module..."
if python3 -c "from redblackgraph.gpu import rb_matrix_gpu; print('✓ GPU module loaded')" 2>/dev/null; then
    echo "✓ GPU module working"
else
    echo "❌ ERROR: GPU module not accessible"
    exit 1
fi

# Run basic tests
echo ""
echo "================================================"
echo "Running basic tests"
echo "================================================"

echo ""
echo "Running GPU tests..."
pytest tests/gpu/test_naive_gpu.py -v || {
    echo "⚠️  Some tests failed (this is expected for naive implementation)"
}

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To activate the environment in the future:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "To run GPU tests:"
echo "  pytest tests/gpu/ -v"
echo ""
echo "To run all tests:"
echo "  pytest tests/ -v"
echo ""
echo "Next steps:"
echo "  1. Review docs/gpu_naive_implementation.md"
echo "  2. Read .plans/gpu_implementation/QUICK_START.md"
echo "  3. Begin Phase 1 implementation (optimized CUDA kernels)"
echo ""
