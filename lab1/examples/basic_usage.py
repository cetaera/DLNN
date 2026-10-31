from jacobi_eigen import jacobi_eigendecomposition


matrix = [
    [4.0, 1.0, -2.0],
    [1.0, 2.0, 0.0],
    [-2.0, 0.0, 3.0],
]

result = jacobi_eigendecomposition(matrix)

print("Eigenvalues:")
for value in result.eigenvalues:
    print(f"  {value:.10f}")

print("Eigenvectors (columns):")
for row in result.eigenvectors:
    print(" ".join(f"{value: .10f}" for value in row))

print(f"Rotations: {result.iterations}")
