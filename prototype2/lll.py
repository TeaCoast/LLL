import numpy as np
import fractions
import math

# 1 - Gram Schmidt Orthogonalization

def compute_GSO(basis: np.ndarray) -> np.ndarray:
    assert len(basis.shape) == 2 and basis.shape[0] == basis.shape[1]
    size = basis.shape[0]

    new_basis = basis.copy()
    for i in range(1, size):
        for j in range(i):
            coefficient = np.dot(basis[i], new_basis[j]) / np.dot(new_basis[j], new_basis[j])
            new_basis[i] -= coefficient * new_basis[j]
    return new_basis

basis = np.array([
    [1, 1, 2],
    [1, 2, 3],
    [2, 1, 3]
], dtype=fractions.Fraction).T

print(compute_GSO(basis).T)

## 1.1 - Gram Schmidt as Fractions?
def simplify_frac(num: int, den: int) -> tuple[int, int]:
    gcd = math.gcd(num, den)
    return num // gcd, den // gcd

def simplify_fracs(nums: np.ndarray, dens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gcds = np.gcd(nums, dens)
    return nums // gcds, dens // gcds

def sum_fracs(nums: np.ndarray, dens: np.ndarray) -> tuple[int, int]:
    den = math.lcm(*dens)
    num = int(sum(den*nums//dens))
    return simplify_frac(num, den)

def inv_frac(num, den):
    return den, num

def neg_frac(num, den):
    return -num, den

def add_2_frac_vecs(nums1: np.ndarray, dens1: np.ndarray, nums2: np.ndarray, dens2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dens = np.lcm(dens1, dens2)
    nums = dens*nums1//dens1 + dens*nums2//dens2
    return simplify_fracs(nums, dens)

def compute_GSO(basis_num: np.ndarray, basis_den: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def generate_coefficient(i: int, j: int) -> tuple[int, int]:
        coef_num_frac = sum_fracs(basis_num[i] * new_basis_num[j], basis_den[i] * new_basis_den[j])
        coef_den_frac = sum_fracs(new_basis_num[j] * new_basis_num[j], new_basis_den[j] * new_basis_den[j])
        coef_num = coef_num_frac[0] * coef_den_frac[1]
        coef_den = coef_num_frac[1] * coef_den_frac[0]
        return simplify_frac(coef_num, coef_den)

    assert len(basis_num.shape) == len(basis_den.shape) == 2
    assert basis_num.shape[0] == basis_num.shape[1] == basis_den.shape[0] == basis_den.shape[1]
    size = basis_num.shape[0]

    new_basis_num = basis_num.copy()
    new_basis_den = basis_den.copy()
    for i in range(1, size):
        for j in range(i):
            coefficient = generate_coefficient(i, j)
            diff_num = -coefficient[0] * new_basis_num[j]
            diff_den = coefficient[1] * new_basis_den[j]
            new_basis_i = add_2_frac_vecs(new_basis_num[i], new_basis_den[i], diff_num, diff_den)
            new_basis_num[i] = new_basis_i[0]
            new_basis_den[i] = new_basis_i[1]
    return new_basis_num, new_basis_den

basis_num = np.array([
    [1, 1, 2],
    [1, 2, 3],
    [2, 1, 3]
]).T

basis_den = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

new_basis = compute_GSO(basis_num, basis_den)

print(new_basis[0].T)
print(new_basis[1].T)

print((new_basis[0] / new_basis[1]).T)