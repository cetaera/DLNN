"""Jacobi eigendecomposition for real symmetric matrices."""

from .solver import (
    ConvergenceError,
    InvalidMatrixError,
    InvalidToleranceError,
    JacobiError,
    JacobiResult,
    NumericalError,
    jacobi_eigendecomposition,
)

__all__ = [
    "ConvergenceError",
    "InvalidMatrixError",
    "InvalidToleranceError",
    "JacobiError",
    "JacobiResult",
    "NumericalError",
    "jacobi_eigendecomposition",
]
