import numpy as np
from numpy.linalg import inv

# Exercise 1
generator = np.random.default_rng(42)
matrix = generator.random((3, 3), dtype=float)
print(matrix)

# Exercise 2
matrix1 = generator.random((3, 3), dtype=float)
matrix2 = generator.random((3, 3), dtype=float)
matrix_sum = matrix1 + matrix2
print(matrix_sum)

# Exercise 3
matrix_product = matrix1.dot(matrix2)
print(matrix_product)

# Exercise 4
matrix_transpose = np.transpose(matrix)
print(matrix_transpose)

# Exercise 5
try:
    matrix_inverse = inv(matrix)
    print(matrix_inverse)
except np.linalg.LinAlgError:
    print("Matrix can not inverse")

det = np.linalg.det(matrix)
print(matrix)

