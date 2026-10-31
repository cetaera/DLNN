"""Spectral decomposition of real symmetric matrices by Jacobi rotations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


class JacobiError(Exception):
    """Base class for errors reported by the package."""

    pass


class InvalidMatrixError(JacobiError, ValueError):
    """The supplied object is not a finite real symmetric matrix."""

    pass


class InvalidToleranceError(JacobiError, ValueError):
    """A tolerance or iteration limit is outside its permitted range."""

    pass


class ConvergenceError(JacobiError, RuntimeError):
    """The requested accuracy was not reached before the iteration limit."""

    pass


class NumericalError(JacobiError, ArithmeticError):
    """The result cannot be represented reliably by finite Python floats."""

    pass


@dataclass(frozen=True)
class JacobiResult:
    """Result of a Jacobi eigendecomposition.

    ``eigenvalues[j]`` corresponds to column ``j`` of ``eigenvectors``.
    ``iterations`` is the number of plane rotations that were performed.
    """

    eigenvalues: Tuple[float, ...]
    eigenvectors: Tuple[Tuple[float, ...], ...]
    iterations: int


def _as_finite_float(value: object, error: type[JacobiError], message: str) -> float:
    if isinstance(value, (bool, str, bytes, complex)):
        raise error(message)
    try:
        converted = float(value)
    except Exception as exc:
        raise error(message) from exc
    if not math.isfinite(converted):
        raise error(message)
    return converted


def _validated_matrix(
    matrix: Iterable[Iterable[object]], symmetry_tolerance: float
) -> list[list[float]]:
    try:
        source_rows = list(matrix)
    except Exception as exc:
        raise InvalidMatrixError("matrix must be an iterable of rows") from exc

    if not source_rows:
        raise InvalidMatrixError("matrix must contain at least one row")

    size = len(source_rows)
    rows: list[list[float]] = []
    for source_row in source_rows:
        try:
            values = list(source_row)
        except Exception as exc:
            raise InvalidMatrixError("every matrix row must be iterable") from exc
        if len(values) != size:
            raise InvalidMatrixError("matrix must be square")
        rows.append(
            [
                _as_finite_float(
                    value,
                    InvalidMatrixError,
                    "matrix entries must be real finite numbers",
                )
                for value in values
            ]
        )

    for row in range(size):
        for column in range(row + 1, size):
            left = rows[row][column]
            right = rows[column][row]
            scale = max(1.0, abs(left), abs(right))
            if abs(left - right) > symmetry_tolerance * scale:
                raise InvalidMatrixError("matrix must be symmetric")
            shared = left * 0.5 + right * 0.5
            rows[row][column] = shared
            rows[column][row] = shared

    return rows


def _identity(size: int) -> list[list[float]]:
    return [
        [1.0 if row == column else 0.0 for column in range(size)]
        for row in range(size)
    ]


def _off_diagonal_norm(matrix: list[list[float]]) -> float:
    """Return the Euclidean norm of the strict upper triangle."""

    result = 0.0
    for row in range(len(matrix) - 1):
        for column in range(row + 1, len(matrix)):
            result = math.hypot(result, matrix[row][column])
    return result


def _rotation_tangent(app: float, aqq: float, apq: float) -> float:
    """Compute the smaller Jacobi rotation without forming a large ratio."""

    delta = aqq - app
    if delta == 0.0:
        return 1.0

    scale = max(abs(delta), abs(apq))
    scaled_delta = delta / scale
    scaled_twice_apq = 2.0 * (apq / scale)
    magnitude = abs(scaled_twice_apq) / (
        abs(scaled_delta) + math.hypot(scaled_delta, scaled_twice_apq)
    )
    same_sign = (delta > 0.0) == (apq > 0.0)
    return magnitude if same_sign else -magnitude


def _rotate(
    matrix: list[list[float]],
    eigenvectors: list[list[float]],
    first: int,
    second: int,
) -> None:
    apq = matrix[first][second]
    tangent = _rotation_tangent(
        matrix[first][first], matrix[second][second], apq
    )
    cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
    sine = tangent * cosine

    app = matrix[first][first]
    aqq = matrix[second][second]
    matrix[first][first] = app - apq * tangent
    matrix[second][second] = aqq + apq * tangent
    matrix[first][second] = 0.0
    matrix[second][first] = 0.0

    for index in range(len(matrix)):
        if index == first or index == second:
            continue
        aip = matrix[index][first]
        aiq = matrix[index][second]
        new_aip = cosine * aip - sine * aiq
        new_aiq = sine * aip + cosine * aiq
        matrix[index][first] = new_aip
        matrix[first][index] = new_aip
        matrix[index][second] = new_aiq
        matrix[second][index] = new_aiq

    for row in eigenvectors:
        vip = row[first]
        viq = row[second]
        row[first] = cosine * vip - sine * viq
        row[second] = sine * vip + cosine * viq

    changed_values = (
        matrix[first][first],
        matrix[second][second],
        *(matrix[index][first] for index in range(len(matrix))),
        *(matrix[index][second] for index in range(len(matrix))),
    )
    if not all(math.isfinite(value) for value in changed_values):
        raise NumericalError("non-finite value produced during a Jacobi rotation")


def _result(
    matrix: list[list[float]],
    eigenvectors: list[list[float]],
    matrix_scale: float,
    iterations: int,
) -> JacobiResult:
    size = len(matrix)
    values = [matrix[index][index] * matrix_scale for index in range(size)]
    if not all(math.isfinite(value) for value in values):
        raise NumericalError("eigenvalues cannot be represented as finite floats")

    for column in range(size):
        norm = 0.0
        for row in range(size):
            norm = math.hypot(norm, eigenvectors[row][column])
        if norm == 0.0 or not math.isfinite(norm):
            raise NumericalError("invalid eigenvector produced by the algorithm")
        for row in range(size):
            eigenvectors[row][column] /= norm

        pivot = max(range(size), key=lambda row: abs(eigenvectors[row][column]))
        if eigenvectors[pivot][column] < 0.0:
            for row in range(size):
                eigenvectors[row][column] = -eigenvectors[row][column]

    order = sorted(range(size), key=values.__getitem__)
    ordered_values = tuple(values[index] for index in order)
    ordered_vectors = tuple(
        tuple(eigenvectors[row][column] for column in order)
        for row in range(size)
    )
    return JacobiResult(ordered_values, ordered_vectors, iterations)


def jacobi_eigendecomposition(
    matrix: Iterable[Iterable[object]],
    tolerance: float = 1e-12,
    max_iterations: Optional[int] = None,
    symmetry_tolerance: float = 1e-12,
) -> JacobiResult:
    """Compute all eigenvalues and eigenvectors of a symmetric matrix.

    Args:
        matrix: Non-empty square iterable of rows. Entries must be finite real
            numbers. The input is copied and is never modified.
        tolerance: Absolute upper bound for the final sum
            ``sum(A[p][q] ** 2 for p != q)``. Must be positive and finite.
        max_iterations: Maximum number of Jacobi rotations. By default the
            limit is ``100 * n * n`` for an ``n`` by ``n`` matrix.
        symmetry_tolerance: Relative tolerance used while checking pairs
            ``A[p][q]`` and ``A[q][p]``. Must be non-negative and finite.

    Returns:
        Eigenvalues in ascending order, the corresponding normalized
        eigenvectors as matrix columns, and the number of rotations.

    Raises:
        InvalidMatrixError: If ``matrix`` violates the input contract.
        InvalidToleranceError: If a tolerance or the iteration limit is
            invalid.
        ConvergenceError: If ``max_iterations`` is reached first.
        NumericalError: If finite floating-point output cannot be produced.

    Repeated eigenvalues may have any orthonormal basis of their eigenspace.
    """

    checked_tolerance = _as_finite_float(
        tolerance,
        InvalidToleranceError,
        "tolerance must be a positive finite number",
    )
    if checked_tolerance <= 0.0:
        raise InvalidToleranceError("tolerance must be a positive finite number")

    checked_symmetry_tolerance = _as_finite_float(
        symmetry_tolerance,
        InvalidToleranceError,
        "symmetry_tolerance must be a non-negative finite number",
    )
    if checked_symmetry_tolerance < 0.0:
        raise InvalidToleranceError(
            "symmetry_tolerance must be a non-negative finite number"
        )

    values = _validated_matrix(matrix, checked_symmetry_tolerance)
    size = len(values)

    if max_iterations is None:
        iteration_limit = max(1, 100 * size * size)
    elif isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise InvalidToleranceError("max_iterations must be a positive integer")
    elif max_iterations <= 0:
        raise InvalidToleranceError("max_iterations must be a positive integer")
    else:
        iteration_limit = max_iterations

    matrix_scale = max(abs(value) for row in values for value in row)
    eigenvectors = _identity(size)
    if matrix_scale == 0.0:
        return _result(values, eigenvectors, 1.0, 0)

    for row in range(size):
        for column in range(size):
            values[row][column] /= matrix_scale

    # _off_diagonal_norm counts one triangle, while the public tolerance is
    # defined for both symmetric halves of the matrix.
    try:
        normalized_threshold = (
            math.sqrt(checked_tolerance) / math.sqrt(2.0) / matrix_scale
        )
    except OverflowError:
        normalized_threshold = math.inf

    residual = _off_diagonal_norm(values)
    if residual <= normalized_threshold:
        return _result(values, eigenvectors, matrix_scale, 0)

    iterations = 0
    while iterations < iteration_limit:
        rotated = False
        limit_reached = False
        for first in range(size - 1):
            for second in range(first + 1, size):
                if iterations >= iteration_limit:
                    limit_reached = True
                    break
                if values[first][second] == 0.0:
                    continue
                _rotate(values, eigenvectors, first, second)
                iterations += 1
                rotated = True
            if limit_reached:
                break

        residual = _off_diagonal_norm(values)
        if residual <= normalized_threshold:
            return _result(values, eigenvectors, matrix_scale, iterations)
        if not rotated:
            raise NumericalError("the residual cannot be reduced further")

    raise ConvergenceError(
        f"Jacobi method did not converge in {iteration_limit} rotations"
    )
