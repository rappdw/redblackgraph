#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' is not installed or not on PATH." >&2
  echo "Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if ! command -v ninja >/dev/null 2>&1; then
  echo "ERROR: 'ninja' is required for meson builds but was not found on PATH." >&2
  echo "On Debian/Ubuntu: sudo apt install ninja-build" >&2
  exit 1
fi

if [[ ! -d "${repo_root}/fs-crawler" ]]; then
  echo "ERROR: Expected submodule directory '${repo_root}/fs-crawler' was not found." >&2
  echo "If you use git submodules, initialize it first:" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi

cd "${repo_root}"

uv venv --seed

rm -rf "${repo_root}/build/cp312"

# Build + test tooling
uv pip install \
  meson-python \
  meson \
  cython \
  tempita \
  numpy \
  pytest \
  pytest-cov \
  pylint

# Per README: use pip directly for editable install with meson-python
PATH="${repo_root}/.venv/bin:${PATH}" .venv/bin/python -m pip install -e . --no-build-isolation

# Optional deps used by some utilities/tests
uv pip install --editable ./fs-crawler XlsxWriter

cat <<'EOF'

Environment is ready.

Next:
  source .venv/bin/activate
  uv run -m pytest

EOF
