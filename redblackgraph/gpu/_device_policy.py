"""
Device policy management for Red-Black Graph GPU operations.

Controls whether operations run on CPU or GPU. Works without CuPy installed
so that code can query the policy even on CPU-only machines.

Environment variable:
    RBG_DEVICE_POLICY  — set to "auto", "cpu", or "gpu" (case-insensitive)

Example usage::

    from redblackgraph.gpu._device_policy import device, DevicePolicy

    # Use env var or explicit setting
    with device(DevicePolicy.GPU):
        result = some_gpu_operation()

    # Query resolved device
    from redblackgraph.gpu._device_policy import resolve_device
    if resolve_device() == DevicePolicy.GPU:
        ...
"""

import enum
import os
import threading
from contextlib import contextmanager


class DevicePolicy(enum.Enum):
    """Device selection policy."""
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


# Thread-local storage for context-manager overrides
_local = threading.local()

# Module-level default (set once via set_device_policy or env var)
_default_policy: DevicePolicy = DevicePolicy.AUTO


def _policy_from_env() -> DevicePolicy:
    """Read device policy from RBG_DEVICE_POLICY environment variable."""
    raw = os.environ.get("RBG_DEVICE_POLICY", "").strip().lower()
    if raw == "cpu":
        return DevicePolicy.CPU
    elif raw == "gpu":
        return DevicePolicy.GPU
    return DevicePolicy.AUTO


# Initialise from env on import
_default_policy = _policy_from_env()


def get_device_policy() -> DevicePolicy:
    """Return the current device policy.

    Checks (in order):
    1. Thread-local override from a ``device()`` context manager
    2. Module-level default set via ``set_device_policy()``
    3. ``RBG_DEVICE_POLICY`` environment variable (read at import time)
    """
    override = getattr(_local, "policy", None)
    if override is not None:
        return override
    return _default_policy


def set_device_policy(policy: DevicePolicy) -> None:
    """Set the module-level default device policy.

    This affects all threads that do not have an active ``device()`` context.
    """
    global _default_policy
    if not isinstance(policy, DevicePolicy):
        raise TypeError(f"Expected DevicePolicy, got {type(policy).__name__}")
    _default_policy = policy


def resolve_device() -> DevicePolicy:
    """Resolve AUTO to a concrete device (CPU or GPU).

    Returns DevicePolicy.GPU if the policy is AUTO and a GPU is available,
    otherwise DevicePolicy.CPU. Explicit CPU/GPU policies are returned as-is.
    """
    policy = get_device_policy()
    if policy != DevicePolicy.AUTO:
        return policy

    # Lazy import to avoid circular dependency and allow this module
    # to be imported even when CuPy is not installed.
    from ._cuda_utils import is_gpu_available
    return DevicePolicy.GPU if is_gpu_available() else DevicePolicy.CPU


@contextmanager
def device(policy: DevicePolicy):
    """Context manager to temporarily override the device policy.

    Args:
        policy: The device policy to use within the context.

    Example::

        with device(DevicePolicy.CPU):
            # All operations in this block use CPU
            ...
    """
    if not isinstance(policy, DevicePolicy):
        raise TypeError(f"Expected DevicePolicy, got {type(policy).__name__}")

    previous = getattr(_local, "policy", None)
    _local.policy = policy
    try:
        yield
    finally:
        _local.policy = previous
