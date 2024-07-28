import numpy as np
from numpy.linalg import inv

generator = np.random.default_rng(42)

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
column_sums = np.einsum("ij ->j", A)
print(A)

# Exercise 2
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
product_a_b = np.einsum("ij,ij->", A, B)
print(product_a_b)

# Exercise 3
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
diagonal_sum = np.einsum("ii->", A)
print(diagonal_sum)

# Exercise 4
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
matrix_multiple = np.einsum("ik,kj->ij", A, B)
print(matrix_multiple)

# Exercise 4
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
matrix_multiple = np.einsum("i,j->ij", A, B)
print(matrix_multiple)


def gram_matrix(tensor):
    channels, height, width = tensor.shape
    features = tensor.reshape(channels, height * width)
    gram = np.einsum("ik,jkl->ij", features, features)

    return gram


tensor1 = generator.random((3, 4, 4), dtype=float)
gram1 = gram_matrix(tensor1)
print(gram1)
