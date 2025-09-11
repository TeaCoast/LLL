# Iterative LLL (ILLL) algorithm

import numpy as np
import lll

PI = int(float("3.1415") * 10000)
E = int(float("2.7181") * 10000)


def A2L(A: np.ndarray, c):
    m, n = A.shape
    lattice = np.zeros(shape=(m+n, m+n))
    for i in range(n):
        lattice[i, i] = 1
    for i in range(n, n+m):
        lattice[i, i] = c
    for A_row, L_row in zip(range(m), range(n, n+m)):
        for col in range(n):
            lattice[L_row][col] = A[A_row][col]
    return lattice

def _main():
    A = np.array([
        [PI, E],
    ])

    c = 0.0001
    L = A2L(A, c)
    
    print(L)

    output = lll.computeLLL(L)
    
    print(output)

if __name__ == "__main__":
    _main()