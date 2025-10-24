import numpy as np
from frac import Frac
from fracs import Fracs
import math

# 1 - gram schmidt with floats
def compute_GSO(basis: np.ndarray) -> np.ndarray:
    """
    input: basis is a square matrix of floats
    output: gso_basis is a square matrix of floats of equal dimension to the input, 
            but each basis is orthogonal to each other, 
            derived from a linear combination of the other basis vectors
    """
    assert len(basis.shape) == 2 and basis.shape[0] == basis.shape[1]
    size = basis.shape[0]

    gso_basis = basis.copy()
    for i in range(1, size):
        for j in range(i):
            coefficient = np.dot(basis[i], gso_basis[j]) / np.dot(gso_basis[j], gso_basis[j])
            gso_basis[i] -= coefficient * gso_basis[j]
    return gso_basis

def compute_GSO_and_coef(basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    input: basis is a square matrix of floats
    output: gso_basis is a square matrix of floats of equal dimension to the input, 
                but each basis is orthogonal to each other, 
                derived from a linear combination of the other basis vectors
            coefs is a lower triangular square matrix with 1 along the diagonals
                with the same shape is theoriginal basis
    """
    assert len(basis.shape) == 2 and basis.shape[0] == basis.shape[1]
    size = basis.shape[0]

    gso_basis = basis.copy()
    coefs = np.identity(size, dtype=float)
    for i in range(1, size):
        for j in range(i):
            coefs[i, j] = np.dot(basis[i], gso_basis[j]) / np.dot(gso_basis[j], gso_basis[j])
            gso_basis[i] -= coefs[i, j] * gso_basis[j]
    return gso_basis, coefs

def update_GSO_and_coef(basis: np.ndarray, gso_basis: np.ndarray, coefs: np.ndarray, start_index: int):
    assert basis.shape == gso_basis.shape == coefs.shape
    assert len(basis.shape) == 2 and basis.shape[0] == basis.shape[1]
    size = basis.shape[0]

    for i in range(start_index, size):
        gso_basis[i] = basis[i]
        for j in range(i):
            coefs[i, j] = np.dot(basis[i], gso_basis[j]) / np.dot(gso_basis[j], gso_basis[j])
            gso_basis[i] -= coefs[i, j] * gso_basis[j]

# 2 - gram schmidt with fractions
def compute_GSO_fracs(basis: Fracs) -> Fracs:
    """
    input: basis is a Fracs that contains 2 square matrices of equal dimention (numerator and denominator)
    output: basis is a Fracs that contains 2 square matrices of equal dimention (numerator and denominator)
    """
    assert len(basis.nums.shape) == 2 and basis.nums.shape[0] == basis.nums.shape[1]
    size = basis.nums.shape[0]

    gso_basis = basis.copy()
    
    for i in range(1, size):
        for j in range(i):
            coefficient = Fracs.dot(basis[i], gso_basis[j]) / Fracs.dot(gso_basis[j], gso_basis[j])
            gso_basis[i] -= coefficient * gso_basis[j]
    return gso_basis


def compute_GSO_and_coef_fracs(basis: Fracs) -> Fracs:
    """
    input: basis is a Fracs object that contains 2 square matrices of equal dimention (numerator and denominator)
    output: basis is a Fracs object that contains 2 square matrices of equal dimention (numerator and denominator)
    """
    assert len(basis.nums.shape) == 2 and basis.nums.shape[0] == basis.nums.shape[1]
    size = basis.nums.shape[0]

    gso_basis = basis.copy()
    coefs = Fracs(np.identity(size, dtype=object))
    for i in range(1, size):
        for j in range(i):
            coefs[i, j] = Fracs.dot(basis[i], gso_basis[j]) / Fracs.dot(gso_basis[j], gso_basis[j])
            gso_basis[i] -= coefs[i, j] * gso_basis[j]
        
        #print(Fracs.dot(basis[i], gso_basis[i]) / Fracs.dot(gso_basis[i], gso_basis[i]))
    return gso_basis, coefs

def update_GSO_and_coef_fracs(basis: Fracs, gso_basis: Fracs, coefs: Fracs, start_index: int):
    assert basis.shape == gso_basis.shape == coefs.shape
    assert len(basis.shape) == 2 and basis.shape[0] == basis.shape[1]
    size = basis.shape[0]

    for i in range(start_index, size):
        gso_basis[i] = basis[i]
        for j in range(i):
            coefs[i, j] = Fracs.dot(basis[i], gso_basis[j]) / Fracs.dot(gso_basis[j], gso_basis[j])
            gso_basis[i] -= coefs[i, j] * gso_basis[j]

def _main():
    basis = np.array([
        [1, 1, 2],
        [1, 2, 3],
        [2, 1, 1]
    ], dtype=object).T

    gso_basis, coefs = compute_GSO_and_coef(basis)

    print(gso_basis.T)
    assert np.array_equal(basis.T, np.dot(gso_basis.T, coefs.T))
    assert np.array_equal(basis, np.dot(coefs, gso_basis))

    basis_fracs = Fracs(basis)

    gso_basis_fracs, coef_fracs = compute_GSO_and_coef_fracs(basis_fracs)

    print(gso_basis_fracs.T)
    assert basis_fracs.T == Fracs.dot(gso_basis_fracs.T, coef_fracs.T)
    assert basis_fracs == Fracs.dot(coef_fracs, gso_basis_fracs)


if __name__ == "__main__":
    _main()