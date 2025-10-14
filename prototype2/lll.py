import numpy as np
import fractions
import math
import gso
from frac import Frac
from fracs import Fracs

# 1 - LLL with floats
def compute_LLL(basis: np.ndarray, alpha: float):
    assert len(basis.shape) == 2 and basis.shape[0] == basis.shape[1]
    assert 1/4 <= alpha <= 1
    size = basis.shape[0]

    gso_basis, coefs = gso.compute_GSO_and_coef(basis)
    LLL_basis = basis.copy()

    def reduce(k, j):
        assert 0 <= j < k < size
        nonlocal gso_basis, coefs
        if abs(coefs[k, j]) > 0.5:
            LLL_basis[k] -= round(coefs[k, j]) * LLL_basis[j]
            gso_basis, coefs = gso.compute_GSO_and_coef(LLL_basis)

    def swap(k, j):
        assert 0 <= k < size and 0 <= j < size
        nonlocal gso_basis, coefs
        LLL_basis[[k, j]] = LLL_basis[[j, k]]
        gso_basis, coefs = gso.compute_GSO_and_coef(LLL_basis)

    k = 1
    while k < size:
        reduce(k, k-1)
        if (np.dot(gso_basis[k-1], gso_basis[k-1])) * (alpha - coefs[k, k-1]**2) > np.dot(gso_basis[k], gso_basis[k]):
            swap(k, k-1)
            k = max(k-1, 1)
        else:
            for j in range(k-2, -1, -1):
                reduce(k, j)
            k += 1
    return LLL_basis

# 1 - LLL with fracs
def compute_LLL_fracs(basis: Fracs, alpha: float):
    assert len(basis.nums.shape) == 2 and basis.nums.shape[0] == basis.nums.shape[1]
    assert 1/4 <= alpha <= 1
    size = basis.nums.shape[0]

    gso_basis, coefs = gso.compute_GSO_and_coef_fracs(basis)
    LLL_basis = basis.copy()

    def reduce(k, j):
        assert 0 <= j < k < size
        nonlocal gso_basis, coefs
        if Frac.abs(coefs[k, j]) > Frac(1, 2):
            LLL_basis[k] -= Frac.round(coefs[k, j]) * LLL_basis[j]
            gso_basis, coefs = gso.compute_GSO_and_coef_fracs(LLL_basis)

    def swap(k, j):
        assert 0 <= k < size and 0 <= j < size
        nonlocal gso_basis, coefs
        temp = LLL_basis[k].copy()
        LLL_basis[k] = LLL_basis[j]
        LLL_basis[j] = temp
        gso_basis, coefs = gso.compute_GSO_and_coef_fracs(LLL_basis)

    k = 1
    while k < size:
        reduce(k, k-1)
        if (Fracs.dot(gso_basis[k-1], gso_basis[k-1])) * (alpha - coefs[k, k-1]**2) > Fracs.dot(gso_basis[k], gso_basis[k]):
            swap(k, k-1)
            k = max(k-1, 1)
        else:
            for j in range(k-2, -1, -1):
                reduce(k, j)
            k += 1
    return LLL_basis

def _main():
    basis = np.array([
        [-2,  7,  7, -5],
        [ 3, -2,  6, -1],
        [ 2, -8, -9, -7],
        [ 8, -9,  6, -4]
    ], dtype=float)
    alpha = 1
    intended_output = np.array([
        [ 2,  3,  1,  1],
        [ 2,  0, -2, -4],
        [-2,  2,  3, -3],
        [ 3, -2,  6, -1]
    ], dtype=float)

    LLL_basis = compute_LLL(basis, alpha)
    assert np.array_equal(LLL_basis, intended_output)

    LLL_basis_fracs = compute_LLL_fracs(Fracs(basis.astype(int)), alpha)
    assert LLL_basis_fracs == Fracs(intended_output.astype(int))


if __name__ == "__main__":
    _main()