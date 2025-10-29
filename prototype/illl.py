# Iterative LLL (ILLL) algorithm

import numpy as np
import lll

PI = float("3.1415")
E = float("2.7181")

# (cI_m, A_mxn)
# {0,   -I_n  )
def A2L(A: np.ndarray, c):
    m, n = A.shape
    lattice = np.zeros(shape=(m+n, m+n))
    for i in range(m):
        lattice[i, i] = c
    for i in range(m, m+n):
        lattice[i, i] = -1
    for row in range(m):
        for A_col, L_col in zip(range(n), range(m, n+m)):
            lattice[row][L_col] = A[row][A_col]
    return lattice

def _main():
    A = np.array([
        [PI, E],
    ])
    m, n = A.shape

    alpha = 3/4
    beta = (4 / (4 * alpha - 1)) # 2
    for delta in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.000001, 0.00000000000000000000000000000000000000000000000000000000000000000000000000000000001]:
        c = beta**(-n*(n+m)/4) * delta**(n+m) # I'm not sure if that n term is n + m - 1

        L = A2L(A, c)

        output = lll.computeLLL(L)
        y1 = output[0]
        q = round(y1[0] / c)
        
        print(f"delta {delta}: q-{q}")

if __name__ == "__main__":
    _main()