"""
Density Monitoring for Sparse Operations

Provides tools to track and warn about matrix density through a processing
pipeline, preventing unexpected O(n²) memory usage.
"""

import warnings
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from ._sparse_format import get_density, get_nnz, is_sparse


class DensificationError(Exception):
    """
    Raised when an operation would cause unacceptable densification.
    
    This exception is raised when matrix density exceeds the configured
    error threshold, typically indicating that a sparse algorithm should
    be used instead of a dense one.
    """
    def __init__(self, message: str, density: float, threshold: float):
        self.density = density
        self.threshold = threshold
        super().__init__(message)


class DensificationWarning(UserWarning):
    """
    Warning issued when matrix density is higher than expected.
    
    This warning indicates that while the operation can proceed,
    the matrix is denser than optimal for sparse operations.
    """
    pass


@dataclass
class DensityRecord:
    """Record of density at a point in the processing pipeline."""
    operation: str
    density: float
    nnz: int
    shape: Tuple[int, int]
    
    def __str__(self) -> str:
        return (f"{self.operation}: density={self.density:.4f}, "
                f"nnz={self.nnz}, shape={self.shape}")


@dataclass
class DensityMonitor:
    """
    Monitor and track matrix density through a processing pipeline.
    
    Use this to ensure operations maintain sparsity and to detect
    unexpected densification that could cause memory issues.
    
    Parameters
    ----------
    warn_threshold : float, default 0.1
        Emit warning when density exceeds this threshold
    error_threshold : float, default 0.5
        Raise error when density exceeds this threshold
    enabled : bool, default True
        Whether monitoring is active
        
    Examples
    --------
    >>> monitor = DensityMonitor(warn_threshold=0.05, error_threshold=0.3)
    >>> monitor.check(A, "after_closure")
    >>> print(monitor.summary())
    """
    warn_threshold: float = 0.1
    error_threshold: float = 0.5
    enabled: bool = True
    history: List[DensityRecord] = field(default_factory=list)
    
    def check(
        self,
        A,
        operation_name: str = "unknown",
        raise_on_error: bool = True,
        warn_on_threshold: bool = True
    ) -> float:
        """
        Check matrix density and record in history.
        
        Parameters
        ----------
        A : array-like or sparse matrix
            Matrix to check
        operation_name : str
            Name of the operation for tracking
        raise_on_error : bool, default True
            Whether to raise DensificationError if error threshold exceeded
        warn_on_threshold : bool, default True
            Whether to emit warning if warn threshold exceeded
            
        Returns
        -------
        float
            Current density
            
        Raises
        ------
        DensificationError
            If density exceeds error_threshold and raise_on_error is True
        """
        if not self.enabled:
            return 0.0
        
        density = get_density(A)
        nnz = get_nnz(A)
        shape = A.shape if hasattr(A, 'shape') else (0, 0)
        
        record = DensityRecord(
            operation=operation_name,
            density=density,
            nnz=nnz,
            shape=shape
        )
        self.history.append(record)
        
        if density > self.error_threshold and raise_on_error:
            raise DensificationError(
                f"Density {density:.4f} exceeds error threshold {self.error_threshold} "
                f"at operation '{operation_name}'. Matrix shape: {shape}, nnz: {nnz}. "
                f"Consider using sparse-preserving algorithms.",
                density=density,
                threshold=self.error_threshold
            )
        
        if density > self.warn_threshold and warn_on_threshold:
            warnings.warn(
                f"Density {density:.4f} exceeds warning threshold {self.warn_threshold} "
                f"at operation '{operation_name}'. Matrix shape: {shape}, nnz: {nnz}.",
                DensificationWarning,
                stacklevel=2
            )
        
        return density
    
    def density_increased(self, threshold: float = 0.01) -> bool:
        """
        Check if density has increased significantly since last check.
        
        Parameters
        ----------
        threshold : float, default 0.01
            Minimum increase to consider significant
            
        Returns
        -------
        bool
            True if density increased by more than threshold
        """
        if len(self.history) < 2:
            return False
        return self.history[-1].density - self.history[-2].density > threshold
    
    def current_density(self) -> Optional[float]:
        """Get the most recent density measurement."""
        if not self.history:
            return None
        return self.history[-1].density
    
    def max_density(self) -> Optional[float]:
        """Get the maximum density observed."""
        if not self.history:
            return None
        return max(r.density for r in self.history)
    
    def summary(self) -> str:
        """
        Generate a summary of density history.
        
        Returns
        -------
        str
            Formatted summary of all density measurements
        """
        if not self.history:
            return "No density measurements recorded."
        
        lines = ["Density History:", "-" * 60]
        for record in self.history:
            status = ""
            if record.density > self.error_threshold:
                status = " [ERROR]"
            elif record.density > self.warn_threshold:
                status = " [WARN]"
            lines.append(f"  {record}{status}")
        
        lines.append("-" * 60)
        lines.append(f"Max density: {self.max_density():.4f}")
        lines.append(f"Final density: {self.current_density():.4f}")
        
        return "\n".join(lines)
    
    def reset(self):
        """Clear the density history."""
        self.history.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - optionally print summary on error."""
        if exc_type is not None and self.history:
            # Print summary on exception for debugging
            print(self.summary())
        return False


# Global default monitor (can be configured by user)
_default_monitor: Optional[DensityMonitor] = None


def get_default_monitor() -> DensityMonitor:
    """Get or create the default density monitor."""
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = DensityMonitor()
    return _default_monitor


def set_default_monitor(monitor: Optional[DensityMonitor]):
    """Set the default density monitor."""
    global _default_monitor
    _default_monitor = monitor


def check_density(A, operation_name: str = "unknown", **kwargs) -> float:
    """
    Convenience function to check density using the default monitor.
    
    Parameters
    ----------
    A : array-like or sparse matrix
        Matrix to check
    operation_name : str
        Name of the operation
    **kwargs
        Additional arguments passed to DensityMonitor.check()
        
    Returns
    -------
    float
        Current density
    """
    return get_default_monitor().check(A, operation_name, **kwargs)


def assert_sparse(A, max_density: float = 0.1, context: str = ""):
    """
    Assert that a matrix is sparse enough for efficient operations.
    
    Parameters
    ----------
    A : array-like or sparse matrix
        Matrix to check
    max_density : float, default 0.1
        Maximum acceptable density
    context : str
        Context string for error message
        
    Raises
    ------
    DensificationError
        If density exceeds max_density
    """
    density = get_density(A)
    if density > max_density:
        ctx = f" ({context})" if context else ""
        raise DensificationError(
            f"Matrix too dense for sparse operation{ctx}: "
            f"density={density:.4f}, max_allowed={max_density}",
            density=density,
            threshold=max_density
        )


def estimate_memory_usage(A, format: str = 'csr') -> int:
    """
    Estimate memory usage of a sparse matrix in bytes.
    
    Parameters
    ----------
    A : sparse matrix or array-like
        Input matrix
    format : str, default 'csr'
        Target format for estimation
        
    Returns
    -------
    int
        Estimated memory usage in bytes
    """
    import numpy as np
    
    if is_sparse(A):
        nnz = A.nnz
        n = max(A.shape)
    else:
        arr = np.asarray(A)
        nnz = np.count_nonzero(arr)
        n = max(arr.shape)
    
    # CSR/CSC: indptr (n+1 ints) + indices (nnz ints) + data (nnz values)
    # Assuming int32 indices and int32 data
    if format in ('csr', 'csc'):
        return (n + 1) * 4 + nnz * 4 + nnz * 4
    elif format == 'coo':
        return nnz * 4 + nnz * 4 + nnz * 4  # row, col, data
    elif format == 'dense':
        return n * n * 4
    else:
        return nnz * 12  # Conservative estimate
