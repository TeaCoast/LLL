# prototype 3
import numpy as np
import math
from gso import GSO
from frac import Frac, Fracs

# 1 - LLL with floats
class LLL:
    basis_in: np.ndarray
    alpha: float

    basis_out: np.ndarray
    transform: np.ndarray
    
    gso: GSO
    gso_basis: np.ndarray
    coefs: np.ndarray
    gso_sq: np.ndarray

    def __init__(self, basis_in: np.ndarray, alpha: float):
        assert len(basis_in.shape) == 2 and basis_in.shape[0] == basis_in.shape[1]
        assert 1/4 <= alpha <= 1
        self.basis_in = basis_in
        self.alpha = alpha

        self.basis_out = basis_in.copy()
        self.transform = np.identity(self.size, dtype=self.dtype)
        
        self.gso = GSO(self.basis_out)
        self.gso_basis = self.gso.basis_out
        self.coefs = self.gso.coefs
        self.gso_sq = np.sum(self.gso_basis**2, axis=1)

        self.compute()

    def reduce_GSO_optimize(self, k: int, j: int, round_kj: int):
        self.coefs[k, j] -= round_kj
        for i in range(j):
            self.coefs[k, i] -= round_kj * self.coefs[j, i]

    def reduce(self, k: int, j: int):
        assert 0 <= j < k < self.size
        if 2 * abs(self.coefs[k, j]) > 1:
            rkj = round(self.coefs[k, j])
            self.basis_out[k] -= rkj * self.basis_out[j]
            self.transform[k] -= rkj * self.transform[j]
            #self.gso.update(j)
            #self.gso_sq[j:] = np.sum(self.gso_basis[j:]**2, axis=1)
            self.reduce_GSO_optimize(k, j, rkj)

    def swap_GSO_opimize(self, k):
        v = self.coefs[k, k-1]
        d = self.gso_sq[k] + v**2 * self.gso_sq[k-1]
        self.coefs[k, k-1] = v * self.gso_sq[k-1] / d
        self.gso_sq[k] *= self.gso_sq[k-1] / d
        self.gso_sq[k-1] = d

        for j in range(k-1):
            self.coefs[k-1, j], self.coefs[k, j] = self.coefs[k, j], self.coefs[k-1, j]
        for i in range(k+1, self.size):
            e = self.coefs[i, k]
            self.coefs[i, k] = self.coefs[i, k-1] - v * e
            self.coefs[i, k-1] = self.coefs[k, k-1] * self.coefs[i, k] + e

    def swap(self, k):
        assert 0 < k < self.size
        self.basis_out[[k, k-1]] = self.basis_out[[k-1, k]]
        self.transform[[k, k-1]] = self.transform[[k-1, k]]
        #self.gso.update(k-1)
        #self.gso_sq[k-1:] = np.sum(self.gso_basis[k-1:]**2, axis=1)
        self.swap_GSO_opimize(k)

    def compute(self):
        k = 1
        while k < self.size:
            self.reduce(k, k-1)
            if self.gso_sq[k-1] * (self.alpha - self.coefs[k, k-1]**2) > self.gso_sq[k]:
                self.swap(k)
                k = max(k-1, 1)
            else:
                for j in range(k-2, -1, -1):
                    self.reduce(k, j)
                k += 1

    @property
    def size(self) -> int:
        return self.basis_in.shape[0]
    @property
    def dtype(self):
        return self.basis_in.dtype


def _main():
    basis = np.array([
        [-2,  7,  7, -5],
        [ 3, -2,  6, -1],
        [ 2, -8, -9, -7],
        [ 8, -9,  6, -4]
    ], dtype=object)
    alpha = 1
    intended_output = np.array([
        [ 2,  3,  1,  1],
        [ 2,  0, -2, -4],
        [-2,  2,  3, -3],
        [ 3, -2,  6, -1]
    ], dtype=object)

    # test with floats
    lll = LLL(basis.astype(float), alpha)
    
    assert np.array_equal(lll.basis_out, intended_output.astype(float))
    assert np.array_equal(lll.basis_out, np.dot(lll.transform, lll.basis_in))

    # test with fracs
    lll = LLL(np.vectorize(Frac)(basis), alpha)

    assert np.array_equal(lll.basis_out, np.vectorize(Frac)(intended_output))
    assert np.array_equal(lll.basis_out, np.dot(lll.transform, lll.basis_in))
    

if __name__ == "__main__":
    _main()