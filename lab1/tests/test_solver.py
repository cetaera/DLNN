import math
import sys
import unittest

from jacobi_eigen import (
    ConvergenceError,
    InvalidMatrixError,
    InvalidToleranceError,
    JacobiResult,
    NumericalError,
    jacobi_eigendecomposition,
)


class SpectralAssertions:
    def assert_decomposition(self, matrix, result, delta=5e-10):
        size = len(matrix)
        self.assertEqual(len(result.eigenvalues), size)
        self.assertEqual(len(result.eigenvectors), size)
        self.assertTrue(all(len(row) == size for row in result.eigenvectors))

        vectors = result.eigenvectors
        for column, eigenvalue in enumerate(result.eigenvalues):
            for row in range(size):
                left = sum(
                    matrix[row][index] * vectors[index][column]
                    for index in range(size)
                )
                right = eigenvalue * vectors[row][column]
                scale = max(1.0, abs(left), abs(right))
                self.assertAlmostEqual(left, right, delta=delta * scale)

        for first in range(size):
            for second in range(size):
                product = sum(
                    vectors[row][first] * vectors[row][second]
                    for row in range(size)
                )
                expected = 1.0 if first == second else 0.0
                self.assertAlmostEqual(product, expected, delta=delta)

        for row in range(size):
            for column in range(size):
                reconstructed = sum(
                    vectors[row][index]
                    * result.eigenvalues[index]
                    * vectors[column][index]
                    for index in range(size)
                )
                scale = max(1.0, abs(matrix[row][column]))
                self.assertAlmostEqual(
                    reconstructed,
                    matrix[row][column],
                    delta=delta * scale,
                )


class JacobiDecompositionTests(SpectralAssertions, unittest.TestCase):
    def test_known_two_by_two_matrix(self):
        result = jacobi_eigendecomposition([[2.0, 1.0], [1.0, 2.0]])

        self.assertIsInstance(result, JacobiResult)
        self.assertEqual(result.iterations, 1)
        self.assertAlmostEqual(result.eigenvalues[0], 1.0, places=14)
        self.assertAlmostEqual(result.eigenvalues[1], 3.0, places=14)
        self.assert_decomposition([[2.0, 1.0], [1.0, 2.0]], result)

    def test_general_symmetric_matrix(self):
        matrix = [
            [4.0, 1.0, -2.0],
            [1.0, 2.0, 0.0],
            [-2.0, 0.0, 3.0],
        ]

        result = jacobi_eigendecomposition(matrix, tolerance=1e-24)

        self.assertEqual(tuple(sorted(result.eigenvalues)), result.eigenvalues)
        self.assert_decomposition(matrix, result)

    def test_diagonal_matrix_is_only_reordered(self):
        matrix = [[7.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 3.0]]

        result = jacobi_eigendecomposition(matrix)

        self.assertEqual(result.eigenvalues, (-2.0, 3.0, 7.0))
        self.assertEqual(result.iterations, 0)
        self.assert_decomposition(matrix, result)

    def test_zero_and_one_by_one_matrices(self):
        zero = jacobi_eigendecomposition([[0.0, 0.0], [0.0, 0.0]])
        scalar = jacobi_eigendecomposition([[-4.5]])

        self.assertEqual(zero.eigenvalues, (0.0, 0.0))
        self.assertEqual(zero.eigenvectors, ((1.0, 0.0), (0.0, 1.0)))
        self.assertEqual(scalar.eigenvalues, (-4.5,))
        self.assertEqual(scalar.eigenvectors, ((1.0,),))

    def test_repeated_eigenvalue(self):
        matrix = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 8.0]]

        result = jacobi_eigendecomposition(matrix)

        self.assertEqual(result.eigenvalues, (3.0, 3.0, 8.0))
        self.assert_decomposition(matrix, result)

    def test_input_is_not_modified(self):
        matrix = [[2.0, -1.0], [-1.0, 4.0]]
        original = [row[:] for row in matrix]

        jacobi_eigendecomposition(matrix)

        self.assertEqual(matrix, original)

    def test_iterators_are_valid_input(self):
        rows = (iter(row) for row in ([2, 1], [1, 2]))

        result = jacobi_eigendecomposition(rows)

        self.assertEqual(result.eigenvalues, (1.0, 3.0))

    def test_nearly_symmetric_input_is_averaged(self):
        matrix = [[2.0, 1.0 + 2e-13], [1.0, 3.0]]

        result = jacobi_eigendecomposition(matrix)

        averaged = [[2.0, 1.0 + 1e-13], [1.0 + 1e-13, 3.0]]
        self.assert_decomposition(averaged, result)

    def test_tolerance_uses_both_off_diagonal_halves(self):
        matrix = [[1.0, 1e-4], [1e-4, 2.0]]

        loose = jacobi_eigendecomposition(matrix, tolerance=3e-8)
        strict = jacobi_eigendecomposition(matrix, tolerance=1e-8)

        self.assertEqual(loose.iterations, 0)
        self.assertEqual(strict.iterations, 1)

    def test_large_finite_scale(self):
        matrix = [[3e150, -1e150], [-1e150, 3e150]]

        result = jacobi_eigendecomposition(matrix, tolerance=1e270)

        self.assertAlmostEqual(result.eigenvalues[0] / 1e150, 2.0, places=13)
        self.assertAlmostEqual(result.eigenvalues[1] / 1e150, 4.0, places=13)
        self.assert_decomposition(matrix, result, delta=1e-12)


class ContractTests(unittest.TestCase):
    def test_invalid_matrices(self):
        invalid = [
            [],
            [[1.0, 2.0]],
            [[1.0, 2.0], [2.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[math.nan]],
            [[math.inf]],
            [[1 + 2j]],
            [[True]],
            [["1"]],
            42,
        ]

        for matrix in invalid:
            with self.subTest(matrix=matrix):
                with self.assertRaises(InvalidMatrixError):
                    jacobi_eigendecomposition(matrix)

    def test_invalid_tolerances(self):
        for tolerance in (0, -1, math.nan, math.inf, True, "small", 1 + 0j):
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(InvalidToleranceError):
                    jacobi_eigendecomposition([[1.0]], tolerance=tolerance)

        for symmetry_tolerance in (-1, math.nan, math.inf, True, "small"):
            with self.subTest(symmetry_tolerance=symmetry_tolerance):
                with self.assertRaises(InvalidToleranceError):
                    jacobi_eigendecomposition(
                        [[1.0]], symmetry_tolerance=symmetry_tolerance
                    )

    def test_invalid_iteration_limits(self):
        for limit in (0, -1, 1.5, True, "10"):
            with self.subTest(limit=limit):
                with self.assertRaises(InvalidToleranceError):
                    jacobi_eigendecomposition([[1.0]], max_iterations=limit)

    def test_iteration_limit_reports_convergence_error(self):
        matrix = [[4.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 2.0]]

        with self.assertRaises(ConvergenceError):
            jacobi_eigendecomposition(
                matrix, tolerance=1e-30, max_iterations=1
            )

    def test_unrepresentable_eigenvalue_reports_numerical_error(self):
        largest = sys.float_info.max
        matrix = [[largest, largest], [largest, largest]]

        with self.assertRaises(NumericalError):
            jacobi_eigendecomposition(matrix)


if __name__ == "__main__":
    unittest.main()
