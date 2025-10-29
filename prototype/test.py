import numpy as np

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

matrix[[0, 1]] = matrix[[1, 0]]

print(matrix)