import numpy as np
import fractions
import math
import gso
from frac import Frac
from fracs import Fracs

# 1 - LLL with floats
class LLL:
    size: int
    basis: np.ndarray
    alpha: float

    gso_basis: np.ndarray
    coefs: np.ndarray

    out_basis: np.ndarray
    transform: np.ndarray

    def __init__(self, basis: np.ndarray, alpha: float):
        assert len(basis.shape) == 2 and basis.shape[0] == basis.shape[1]
        assert 1/4 <= alpha <= 1
        self.size = basis.shape[0]
        self.basis = basis
        self.alpha = alpha

        self.gso_basis, self.coefs = gso.compute_GSO_and_coef(basis)
        self.gso_sq = np.sum(self.gso_basis**2, axis=1)
        
        self.out_basis = basis.copy()
        self.transform = np.identity(self.size)

    def reduce_GSO_optimize(self, k, j, round_kj):
        self.coefs[k, j] -= round_kj
        for i in range(j):
            self.coefs[k, i] -= round_kj * self.coefs[j, i]

    def reduce(self, k, j):
        assert 0 <= j < k < self.size
        if abs(self.coefs[k, j]) > 0.5:
            rkj = round(self.coefs[k, j])
            self.out_basis[k] -= rkj * self.out_basis[j]
            self.transform[k] -= rkj * self.transform[j]
            #gso.update_GSO_and_coef(LLL_basis, gso_basis, coefs, j)
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
        self.out_basis[[k, k-1]] = self.out_basis[[k-1, k]]
        self.transform[[k, k-1]] = self.transform[[k-1, k]]
        #gso.update_GSO_and_coef(LLL_basis, gso_basis, coefs, k-1)
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
    
def compute_LLL(basis: np.ndarray, alpha: float):
    assert len(basis.shape) == 2 and basis.shape[0] == basis.shape[1]
    assert 1/4 <= alpha <= 1
    size = basis.shape[0]

    gso_basis, coefs = gso.compute_GSO_and_coef(basis)
    gso_sq = np.sum(gso_basis**2, axis=1)
    LLL_basis = basis.copy()

    def reduce(k, j):
        assert 0 <= j < k < size
        if abs(coefs[k, j]) > 0.5:
            rkj = round(coefs[k, j])
            LLL_basis[k] -= rkj * LLL_basis[j]
            # optimized GSO update for reduction
            coefs[k, j] -= rkj
            for i in range(j):
                coefs[k, i] -= rkj * coefs[j, i]

    def swap(k):
        assert 0 < k < size
        LLL_basis[[k, k-1]] = LLL_basis[[k-1, k]]
        new_gso, new_coefs = gso.compute_GSO_and_coef(LLL_basis)
        # optimized GSO update for swap
        v = coefs[k, k-1]
        d = gso_sq[k] + v**2 * gso_sq[k-1]
        coefs[k, k-1] = v * gso_sq[k-1] / d
        gso_sq[k] *= gso_sq[k-1] / d
        gso_sq[k-1] = d

        for j in range(k-1):
            coefs[k-1, j], coefs[k, j] = coefs[k, j], coefs[k-1, j]
        for i in range(k+1, size):
            e = coefs[i, k]
            coefs[i, k] = coefs[i, k-1] - v * e
            coefs[i, k-1] = coefs[k, k-1] * coefs[i, k] + e

    k = 1
    while k < size:
        reduce(k, k-1)
        if gso_sq[k-1] * (alpha - coefs[k, k-1]**2) > gso_sq[k]:
            swap(k)
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
    print(gso_basis.nums.dtype, gso_basis.dens.dtype)
    gso_sq = gso_basis.__pow__(2).get_row_sums()

    LLL_basis = basis.copy()

    def reduce(k, j):
        assert 0 <= j < k < size
        rkj = round(coefs[k, j])
        if abs(coefs[k, j]) > Frac(1, 2):
            LLL_basis[k] -= rkj * LLL_basis[j]
            #gso.update_GSO_and_coef_fracs(LLL_basis, gso_basis, coefs, j)
            # optimized GSO update for reduction
            coefs[k, j] -= rkj
            for i in range(j):
                coefs[k, i] -= rkj * coefs[j, i]

    def swap(k):
        assert 0 < k < size
        LLL_basis[k], LLL_basis[k-1] = LLL_basis[k-1], LLL_basis[k].copy()
        v = coefs[k, k-1]
        d = gso_sq[k] + v**2 * gso_sq[k-1]
        coefs[k, k-1] = v * gso_sq[k-1] / d
        gso_sq[k] = gso_sq[k] * gso_sq[k-1] / d
        gso_sq[k-1] = d

        for j in range(k-1):
            coefs[k-1, j], coefs[k, j] = coefs[k, j], coefs[k-1, j]
        for i in range(k+1, size):
            e = coefs[i, k]
            coefs[i, k] = coefs[i, k-1] - v * e
            coefs[i, k-1] = coefs[k, k-1] * coefs[i, k] + e
            
    k = 1
    while k < size:
        reduce(k, k-1)
        if gso_sq[k-1] * (alpha - coefs[k, k-1]**2) > gso_sq[k]:
            swap(k)
            k = max(k-1, 1)
        else:
            for j in range(k-2, -1, -1):
                reduce(k, j)
            k += 1
    return LLL_basis

class LLL_Fracs:
    size: int
    basis: Fracs
    alpha: float

    gso_basis: Fracs
    coefs: Fracs

    out_basis: Fracs
    transform: Fracs

    def __init__(self, basis: Fracs, alpha: float):
        assert len(basis.shape) == 2 and basis.shape[0] == basis.shape[1]
        assert 1/4 <= alpha <= 1
        self.size = basis.shape[0]
        self.basis = basis
        self.alpha = alpha

        self.gso_basis, self.coefs = gso.compute_GSO_and_coef_fracs(basis)
        self.gso_sq = self.gso_basis.get_row_sums()
        
        self.out_basis = basis.copy()
        self.transform = Fracs(np.identity(self.size))

    def reduce_GSO_optimize(self, k, j, round_kj):
        self.coefs[k, j] -= round_kj
        for i in range(j):
            self.coefs[k, i] -= round_kj * self.coefs[j, i]

    def reduce(self, k, j):
        assert 0 <= j < k < self.size
        if abs(self.coefs[k, j]) > 0.5:
            rkj = round(self.coefs[k, j])
            self.out_basis[k] -= rkj * self.out_basis[j]
            self.transform[k] -= rkj * self.transform[j]
            #gso.update_GSO_and_coef(LLL_basis, gso_basis, coefs, j)
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
        self.out_basis[k], self.out_basis[k-1] = self.out_basis[k-1], self.out_basis[k].copy()
        self.transform[k], self.transform[k-1] = self.transform[k-1], self.transform[k].copy()
        #gso.update_GSO_and_coef(LLL_basis, gso_basis, coefs, k-1)
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

    lll = LLL(basis, alpha)
    lll.compute()

    LLL_basis = compute_LLL(basis, alpha)
    frac_basis = Fracs(basis)
    print(frac_basis.nums.dtype, frac_basis.dens.dtype)
    LLL_basis_fracs = compute_LLL_fracs(Fracs(basis), alpha)

    assert np.array_equal(LLL_basis, np.dot(lll.transform, lll.basis))

    print(LLL_basis_fracs)

    assert np.array_equal(LLL_basis, intended_output)
    assert LLL_basis_fracs == Fracs(intended_output)

if __name__ == "__main__":
    _main()