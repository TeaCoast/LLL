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

def _main():
    basis = np.array([
        [1, 1, 2],
        [1, 2, 3],
        [2, 1, 1]
    ], dtype=float).T

    gso_basis = compute_GSO(basis)

    print(gso_basis.T)

    basis_fracs = Fracs(basis.astype(int))

    gso_basis_fracs = compute_GSO_fracs(basis_fracs)

    print(gso_basis_fracs.T)

if __name__ == "__main__":
    _main()