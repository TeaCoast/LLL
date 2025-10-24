import math
import numpy as np
from frac import Frac

class Fracs:
    nums: np.ndarray
    dens: np.ndarray
    
    def __init__(self, nums: np.ndarray, dens: np.ndarray = None):
        if dens is None:
            dens = np.ones(nums.shape, dtype=object)
        assert nums.shape == dens.shape
        assert 0 not in dens
        self.nums = nums
        self.dens = dens
        self.simplify()
        
    def simplify(self):
        GCDs = np.gcd(self.nums, self.dens, dtype=object)
        self.nums //= GCDs
        self.dens //= GCDs
        self.nums *= np.sign(self.dens)
        self.dens = np.abs(self.dens)
        return self

    def get_sum(self) -> Frac:
        den = math.lcm(*self.dens.flatten())
        num = sum(den*self.nums.flatten()//self.dens.flatten())
        return Frac(int(num), den)
    
    def get_prod(self) -> Frac:
        den = np.prod(self.dens.flatten())
        num = np.prod(self.nums.flatten())
        return Frac(int(num), int(den))
    
    def dot(fracs1: 'Fracs', fracs2: 'Fracs') -> Frac:
        if len(fracs1.nums.shape) == len(fracs2.nums.shape) == 1:
            assert fracs1.nums.shape == fracs2.nums.shape
            return fracs1.__mul__(fracs2).get_sum()
        elif len(fracs1.nums.shape) == len(fracs2.nums.shape) == 2:
            assert fracs1.nums.shape[1] == fracs2.nums.shape[0]
            rows, cols = (fracs1.nums.shape[0], fracs2.nums.shape[1])
            new_fracs = Fracs(np.zeros((rows, cols), dtype=object))
            for col in range(cols):
                col_fracs = fracs2.T[col]
                for row in range(cols):
                    new_fracs[row, col] = fracs1[row].dot(col_fracs)
            return new_fracs
        raise ValueError("shapes of fracs1 and fracs2 do not align for dot product")

    def get_row_sums(self) -> 'Fracs':
        assert len(self.shape) == 2
        dens = np.apply_along_axis(lambda array: np.lcm.reduce(array, dtype=object), axis=1, arr=self.dens)
        nums = np.sum(dens[:, np.newaxis]*self.nums//self.dens, axis=1, dtype=object)
        return Fracs(nums, dens)
    
    def copy(self) -> 'Fracs':
        return Fracs(self.nums.copy(), self.dens.copy())

    def __neg__(fracs: 'Fracs') -> 'Fracs':
        return Fracs(-fracs.nums, fracs.dens)

    def __add__(fracs1: 'Fracs', fracs2: 'Fracs') -> 'Fracs':
        if isinstance(fracs2, Fracs):
            nums2 = fracs2.nums
            dens2 = fracs2.dens
        else:
            try: nums2, dens2 = Frac(fracs2)
            except TypeError: return NotImplemented
        dens = np.lcm(fracs1.dens, dens2)
        nums = dens*fracs1.nums//fracs1.dens + dens*nums2//dens2
        fracs = Fracs(nums, dens)
        fracs.simplify()
        return fracs
    
    def __radd__(fracs1: 'Fracs', frac2: int | tuple[int, int] | Frac) -> 'Fracs':
        return fracs1.__add__(frac2)

    def __sub__(fracs1: 'Fracs', fracs2: 'Fracs') -> 'Fracs':
        if isinstance(fracs2, Fracs):
            nums2 = fracs2.nums
            dens2 = fracs2.dens
        else:
            try: nums2, dens2 = Frac(fracs2)
            except TypeError: return NotImplemented
        dens = np.lcm(fracs1.dens, dens2)
        nums = dens*fracs1.nums//fracs1.dens - dens*nums2//dens2
        return Fracs(nums, dens).simplify()
    
    def __rsub__(fracs1: 'Fracs', frac2: int | tuple[int, int] | Frac) -> 'Fracs':
        return fracs1.__neg__().__add__(frac2)
    
    def inv(self) -> 'Fracs':
        return Fracs(self.dens, self.nums)

    def __mul__(fracs1: 'Fracs', fracs2: 'Fracs') -> 'Fracs':
        if isinstance(fracs2, Fracs):
            nums2 = fracs2.nums
            dens2 = fracs2.dens
        else:
            try: nums2, dens2 = Frac(fracs2)
            except TypeError: return NotImplemented
        dens = fracs1.dens * dens2
        nums = fracs1.nums * nums2
        return Fracs(nums, dens).simplify()
    
    def __rmul__(fracs1: 'Fracs', fracs2: 'Fracs') -> 'Fracs':
        return fracs1.__mul__(fracs2)
    
    def __truediv__(fracs1: 'Fracs', fracs2: 'Fracs') -> 'Fracs':
        if isinstance(fracs2, Fracs):
            nums2 = fracs2.nums
            dens2 = fracs2.dens
        else:
            try: nums2, dens2 = Frac(fracs2)
            except TypeError: return NotImplemented
        dens = fracs1.dens * nums2
        nums = fracs1.nums * dens2
        return Fracs(nums, dens).simplify()
        
    def __rtruediv__(fracs1: 'Fracs', fracs2: 'Fracs') -> 'Fracs':
        return fracs1.inv().__mul__(fracs2)
    
    def __pow__(fracs1: 'Fracs', power: int) -> 'Fracs':
        if not isinstance(power, (int, np.signedinteger)):
            return NotImplemented
        if power < 0:
            return Fracs(fracs1.dens ** power, fracs1.nums ** power).simplify()
        return Fracs(fracs1.nums ** power, fracs1.dens ** power).simplify()

    def __eq__(fracs1: 'Fracs', fracs2: 'Fracs') -> bool:
        if not isinstance(fracs1, Fracs) or not isinstance(fracs2, Fracs):
            return NotImplemented
        return np.array_equal(fracs1.nums, fracs2.nums) and np.array_equal(fracs1.dens, fracs2.dens)

    def __getitem__(self, index: int | tuple[int, ...]) -> 'Frac | Fracs':
        nums = self.nums[index]
        dens = self.dens[index]
        if isinstance(nums, (int, np.signedinteger)):
            return Frac(int(nums), int(dens))
        return Fracs(nums, dens)
    
    def __setitem__(self, index: int | tuple[int, ...], value: 'Frac | Fracs'):
        if isinstance(index, int):
            index = (index, )
        assert isinstance(index, tuple)
        if isinstance(value, Frac ):
            assert len(index) == len(self.nums.shape)
            self.nums[index] = value.num
            self.dens[index] = value.den
        elif isinstance(value, Fracs):
            assert len(index) < len(self.nums.shape)
            assert value.nums.shape == self.nums.shape[len(index):]
            self.nums[index] = value.nums
            self.dens[index] = value.dens
        else:
            raise TypeError("value can only by Frac or Fracs")
    
    @property
    def shape(self) -> tuple[int, ...]:
        return self.nums.shape

    @property
    def T(self) -> 'Fracs':
        return Fracs(self.nums.T, self.dens.T)

    def __str__(self):
        array: np.ndarray = self.nums.astype(str) + '/' + self.dens.astype(str)
        return str(array).replace("'", "")
    

def _main():
    hi = Fracs(np.array([5, 3, 2]), np.array([7, 5, 3]))
    hello = Fracs(np.array([2, 7, 9]), np.array([8, 1, 3]))
    assert Fracs(np.array([27, 38, 11]), np.array([28, 5, 3])) == hi + hello == hello + hi
    assert Fracs(np.array([13, -32, -7]), np.array([28, 5, 3])) == hi - hello == -hello + hi
    assert Fracs(np.array([12, 8, 5]), np.array([7, 5, 3])) == hi + 1 == 1 + hi
    assert Fracs(np.array([25, 3, 5]), np.array([28, 4, 6])) == hi * Frac(5, 4) == Frac(5, 4) * hi == hi / Frac(4, 5)
    assert Fracs(np.array([8, 2, 2]), np.array([3, 21, 9])) == Frac(2, 3) / hello

    matrix = Fracs(np.array([[1, 2], [3, 4]]), np.array([[9, 8], [7, 6]]))
    assert matrix == matrix
    assert matrix[0] == Fracs(np.array([1, 2]), np.array([9, 8]))
    assert matrix[0, 1] == Frac(1, 4)
    assert matrix[0].dot(matrix[1]) == Frac(3, 14)
    assert matrix.get_row_sums() == Fracs(np.array([13, 23]), np.array([36, 21]))
    assert matrix ** 2 == Fracs(np.array([[1, 1], [9, 4]]), np.array([[81, 16], [49, 9]]))
    matrix[0, 1] = Frac(9, 2)
    assert matrix == Fracs(np.array([[1, 9], [3, 2]]), np.array([[9, 2], [7, 3]]))
    matrix[0] = Fracs(np.array([5, 3]), np.array([7, 8]))
    assert matrix == Fracs(np.array([[5, 3], [3, 2]]), np.array([[7, 8], [7, 3]]))

if __name__ == "__main__":
    _main()