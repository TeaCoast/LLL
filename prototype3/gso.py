# prototype 3
from frac import Frac, Fracs
import numpy as np

class GSO:
    """
    basis (np.ndarray) - input, square matrix
    gso_basis (np.ndarray) - output, square matrix
        - orthogonalized basis derived from basis
    coefs (np.ndarray) - output, square matrix, 
        - lower triangular (row form) with 1 on the diagonals
        - transformation matrix from gso_basis to basis (basis = coefs * gso_basis)
    """
    
    basis_in: np.ndarray
    basis_out: np.ndarray
    coefs: np.ndarray
    
    def __init__(self, basis_in: np.ndarray):
        assert len(basis_in.shape) == 2 and basis_in.shape[0] == basis_in.shape[1]
        self.basis_in = basis_in
        self.basis_out = np.zeros(shape=(self.size, self.size), dtype=self.dtype)
        self.coefs = np.identity(self.size, dtype=self.dtype)
        self.update()
    
    def update(self, start_index: int = 0):
        """computes gso_basis and coefs beginning from a start index )"""
        assert 0 <= start_index < self.size
        self.basis_out[start_index:] = self.basis_in[start_index:]
        for i in range(start_index, self.size):
            for j in range(i):
                self.coefs[i, j] = np.dot(self.basis_in[i], self.basis_out[j]) / np.dot(self.basis_out[j], self.basis_out[j])
                self.basis_out[i] -= self.coefs[i, j] * self.basis_out[j]

    @property
    def size(self) -> int:
        return self.basis_in.shape[0]
    @property
    def dtype(self):
        return self.basis_in.dtype

def _main():
    basis = np.array([
        [1, 1, 2000],
        [1000, 2, 3],
        [2, 10000, 1]
    ], dtype=float).T

    
    gso = GSO(basis)
    assert np.all(np.abs(gso.basis_in.T - np.dot(gso.basis_out.T, gso.coefs.T)) < 0.00000001)
    assert np.all(np.abs(gso.basis_in - np.dot(gso.coefs, gso.basis_out)) < 0.00000001)

    
    basis_frac = Fracs([
        [300,     2000,     20000000],
        [1000000, 20,       3],
        [2,       10000000, 10]
    ]).T

    gso = GSO(basis_frac)
    assert np.array_equal(gso.basis_in.T, np.dot(gso.basis_out.T, gso.coefs.T))
    assert np.array_equal(gso.basis_in, np.dot(gso.coefs, gso.basis_out))
    for i in range(gso.size):
        for j in range(i):
            assert np.dot(gso.basis_out[i], gso.basis_out[j]) == 0



if __name__ == "__main__":
    _main()