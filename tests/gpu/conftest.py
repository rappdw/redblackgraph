"""Shared fixtures and markers for GPU tests."""

import pytest

try:
    from redblackgraph.gpu._cuda_utils import CUPY_AVAILABLE
except ImportError:
    CUPY_AVAILABLE = False


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked with @pytest.mark.gpu when CuPy is unavailable."""
    if CUPY_AVAILABLE:
        return
    skip_gpu = pytest.mark.skip(reason="CuPy / CUDA GPU not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture
def requires_gpu():
    """Fixture that skips the test if CuPy is not available."""
    if not CUPY_AVAILABLE:
        pytest.skip("CuPy / CUDA GPU not available")
