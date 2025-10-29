# LLL algorithm

import numpy as np

# matrix standard
"""
each row is a bsis vector
[a b c]
[d e f]
[g h i]
"""

def get_coef_index(u_index, v_index):
    """
    Converts indices of 2D pyramid matrix into linear index
            v
         0 1 2 3 4   
      0 | | | | | |
      1 |0| | | | |
    u 3 |1|2| | | |
      4 |3|4|5| | |
      5 |6|7|8|9| |
    """
    if v_index >= u_index:
        return None
    return u_index*(u_index - 1) // 2 + v_index 

def computeGSO(basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes gram schmidt orthogonalization on a matrix of basis vectors
    returns both the orthogonalized basis and its coefficients
    """
    basis_ortho = basis.copy()

    vec_count, vec_length = basis.shape

    coefficients = []

    for i in range(1, vec_count):
        for j in range(i):
            u: np.ndarray = basis_ortho[i]
            v: np.ndarray = basis_ortho[j]
            coefficients.append(u.dot(v) / v.dot(v))
            basis_ortho[i] -= v * coefficients[-1]
    return basis_ortho, coefficients


def computeLLL(basis: np.ndarray, a: float=3/4) -> np.ndarray:
    """
    Computes the LLL alpha reduced basis for the Latice based on the matri of basis vectors
    returns the alpha reduced basis
    """

    basis_reduced = basis.copy()

    basis_ortho, coefficients = computeGSO(basis)

    vec_count, vec_length = basis.shape

    gamma = np.array([np.dot(basis_ortho[i], basis_ortho[i]) for i in range(vec_count)])

    def get_coef(u, v):
        return coefficients[get_coef_index(u, v)]
    
    def set_coef(u, v, value):
        coefficients[get_coef_index(u, v)] = value

    def reduce(k, l):
        coef_kl = get_coef(k, l)
        if abs(coef_kl) > 1/2:
            # apply rounded gram schmidt operation
            basis_reduced[k] -= round(coef_kl) * basis_reduced[l]
            # alter gram schmidt coefficients
            for j in range(l):
                coefficients[get_coef_index(k, j)] -= round(coef_kl) * get_coef(l, j)
            coefficients[get_coef_index(k, l)] -= round(coef_kl)

    def exchange(k):
        # swap k with k - 1
        basis_reduced[[k, k-1]] = basis_reduced[[k-1, k]]

        v = get_coef(k, k-1)
        d = gamma[k] + v**2 * gamma[k-1]
        set_coef(k, k-1, v * gamma[k-1] / d)
        gamma[k] *= gamma[k-1] / d
        gamma[k-1] = d

        for j in range(k-1):
            # swap coef_k-1,j and coef_kj
            t = get_coef(k-1, j)
            set_coef(k-1, j, get_coef(k, j))
            set_coef(k, j, t)
        
        for i in range(k+1, vec_count):
            e = get_coef(i, k)
            set_coef(i, k, get_coef(i, k-1) - v * get_coef(i, k))
            set_coef(i, k-1, get_coef(k, k-1) * get_coef(i, k) + e)

    k = 1
    while k < vec_count:
        reduce(k, k-1)
        if gamma[k] >= (a - get_coef(k, k-1) ** 2) * gamma[k-1]:
            for l in range(k-2, -1, -1):
                reduce(k, l)
            k += 1
        else:
            exchange(k)
            if k > 1:
                k -= 1
    return basis_reduced

def _main():
    basis = np.array([
        [-2,  7,  7, -5],
        [ 3, -2,  6, -1],
        [ 2, -8, -9, -7],
        [ 8, -9,  6, -4]
    ], dtype=float)

    basis_reduced = computeLLL(basis, 1)

    print(basis_reduced)

if __name__ == "__main__":
    _main()