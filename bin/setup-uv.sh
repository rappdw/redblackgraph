#!/usr/bin/env bash
set -euo pipefail

# Setup development environment using uv.
#
# Usage:
#   ./bin/setup-uv.sh          # CPU only
#   ./bin/setup-uv.sh --gpu    # Include GPU (CuPy) dependencies
#
# Prerequisites:
#   - uv on PATH (https://docs.astral.sh/uv/)
#   - fs-crawler submodule initialized (git submodule update --init)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' is not installed or not on PATH." >&2
  echo "Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [[ ! -d "${repo_root}/fs-crawler" ]]; then
  echo "Initializing fs-crawler submodule..."
  git submodule update --init --recursive
fi

install_gpu=0
if [[ "${1:-}" == "--gpu" ]]; then
  install_gpu=1
fi

# Create venv with uv-managed Python
uv venv --seed

# Clean stale meson build dirs (editable loader references a specific path)
rm -rf "${repo_root}/build"

# Install build dependencies into the venv.
# Meson-python editable installs require --no-build-isolation, which means
# build deps from [build-system].requires must be pre-installed.
uv pip install \
  meson-python \
  meson \
  ninja \
  cython \
  tempita \
  numpy

# Editable install with test + io extras
uv pip install -e ".[test,io]" --no-build-isolation

# GPU extras (optional)
if [[ "${install_gpu}" == "1" ]]; then
  uv pip install cupy-cuda12x
fi

cat <<'EOF'

Environment is ready.

  source .venv/bin/activate
  pytest tests/

EOF
