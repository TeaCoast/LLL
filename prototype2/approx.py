import math
import numpy as np
from frac import Frac
from fracs import Fracs
import lll


# Block form:
# A = (I  0)
#     (A cI)
def generate_basis(alpha_matrix: np.ndarray, delta: float, beta: float = 2):
    assert len(alpha_matrix.shape) == 2
    assert 0 < delta < 1
    assert beta > 4/3
    m, n = alpha_matrix.shape
    size = n + m

    c = (beta**(-(n+m-1))*delta**4) ** ((n + m) / (4m))
    B = np.identity(size, dtype=float)
    for row in range(m):
        for col in range(n):
            B[row+n, col] = alpha_matrix[row, col]
        B[row+n, row+n] = c
    return B

def get_num(number: float, den: int | np.ndarray):
    return np.round(number*den)

def generate_approx(alpha_matrix: np.ndarray, delta: float, alpha: float = 3/4): 
    assert 1/4 < alpha <= 1
    beta = 1 / (alpha - 1/4)
    basis = generate_basis(alpha_matrix, delta, beta)
    m, n = alpha_matrix.shape
    c = basis[n, n]

    lll_basis = lll.compute_LLL(basis, alpha)
    y1 = lll_basis[0]
    q_list: np.ndarray = y1[n:] / c
    return q_list.astype(int)

def ill(alpha_matrix: np.ndarray, delta: float, q_max: int, alpha: float = 3/4):
    assert 1/4 < alpha <= 1
    beta = 1 / (alpha - 1/4)
    basis = generate_basis(alpha_matrix, delta, beta)
    m, n = alpha_matrix.shape
    c = basis[n, n]
    k = math.ceil(m/n * math.log2(q_max) -(m+n-1)*(m+n)/(4*n))
    
    d = 1 / delta

    for i in range(k):
        lll_basis = lll.compute_LLL(basis, alpha)
        y1 = lll_basis[0]
        q_list: np.ndarray = y1[n:] / c
        c *= d**((m+n)/n)
        for i in range(m):
            basis[n + i, n + i] = c
    return q_list.astype(int)

def _main():
    alpha_matrix = np.array([
        [3.25],
        [1.81]
    ]).T

    basis = generate_basis(alpha_matrix, 0.5)
    assert np.array_equal(basis.T, np.array([
        [1, 0, 3.25],
        [0, 1, 1.81],
        [0, 0, 8   ]
    ], dtype=float))

    alpha_matrix = np.array([
        [math.pi],
        [math.e]
    ]).T

    delta = 0.00000001
    q_list = generate_approx(alpha_matrix, delta)
    den = q_list[0]
    p_list = get_num(alpha_matrix[0], den)
    assert abs(math.pi * den - p_list[0]) <= delta
    assert abs(math.e * den - p_list[1]) <= delta
    
    
if __name__ == "__main__":
    _main()