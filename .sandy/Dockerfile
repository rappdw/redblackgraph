# .sandy/Dockerfile — GPU support for sandy
#
# Copy this file to your project's .sandy/Dockerfile:
#   cp examples/gpu/Dockerfile .sandy/Dockerfile
#
# Then enable GPU passthrough in .sandy/config:
#   SANDY_GPU=all
#
# Requires NVIDIA Container Toolkit on the host:
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
#
# Works on x86_64 and arm64 (including DGX Spark).

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# --- Option 1: Full CUDA toolkit (needed for compiling CUDA code) ---
# Adds ~3-5GB. Use this if you need nvcc, cuDNN headers, or are building
# custom CUDA kernels.
#
# The repo's Debian release must match the base image (sandy-base is Debian 13
# trixie). The debian12 repo fails here: its signing key relies on a SHA-1
# binding signature, which trixie's apt (sqv) rejects since 2026-02-01.

RUN CUDA_ARCH="$(uname -m)"; [ "$CUDA_ARCH" = "aarch64" ] && CUDA_ARCH="sbsa"; \
    curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/debian13/${CUDA_ARCH}/cuda-keyring_1.1-1_all.deb" \
        -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

# --- Option 2: PyTorch only (lighter, no system CUDA needed) ---
# Uncomment below and comment out Option 1 if you just need PyTorch.
# Pre-built wheels include their own CUDA runtime (~2GB download).

# RUN pip install --user torch torchvision torchaudio

# --- CuPy (GPU-accelerated NumPy, used by redblackgraph GPU backend) ---
# Installed system-wide: build steps run as root, so --user would land in
# /root/.local, invisible to the sandy user at runtime. cupy-cuda13x matches
# the CUDA 13 toolkit above; if no wheel exists for this platform, fall back
# to building 'cupy' from source against that toolkit.
RUN pip install --no-cache-dir --break-system-packages cupy-cuda13x \
    || pip install --no-cache-dir --break-system-packages cupy

# --- Python ML packages (persist across sessions via pip sandbox) ---
# Uncomment what you need:

# RUN pip install --user transformers accelerate
# RUN pip install --user jupyter
# RUN pip install --user numpy pandas scikit-learn

