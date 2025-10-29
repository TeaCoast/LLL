# prototype 3
import math
import numpy as np
from dataclasses import dataclass, field

def _simplify(frac: tuple[int, int]):
    num, den = frac
    num *= int(np.sign(den))
    den = abs(den)
    gcd = math.gcd(num, den)
    return (num // gcd, den // gcd)


class Frac:
    num: int
    den: int = 1

    def __new__(cls, num: "int | Frac | tuple[int, int]", den: int = 1):
        try:
            if isinstance(num, Frac):
                assert den == 1
                return num
            elif isinstance(num, tuple):
                assert len(num) == 2
                assert den == 1
                num, den = num
            # TODO - implement float behavior
            assert isinstance(num, int)
            assert isinstance(den, int)
        except AssertionError:
            raise TypeError
        self = super().__new__(cls)
        self.num, self.den = _simplify((num, den))
        return self

    def inv(frac: 'Frac') -> 'Frac':
        try:
            num, den = Frac(frac)
        except TypeError:
            return NotImplemented
        return Frac((den*int(np.sign(num)), abs(num)))
    
    def floordiv(frac: 'Frac') -> int:
        return frac.num // frac.den

    def floatdiv(frac: 'Frac') -> float:
        return frac.num / frac.den

    def __round__(frac: 'Frac') -> int:
        return frac.__add__(Frac(1, 2)).floordiv()
    
    def __abs__(frac: 'Frac') -> 'Frac':
        return Frac(abs(frac.num), frac.den)
    
    def __neg__(frac: 'Frac') -> 'Frac':
        try:
            num, den = Frac(frac)
        except TypeError:
            return NotImplemented
        return Frac((-num, den))

    def __add__(frac1: 'Frac', frac2: 'Frac'):
        try:
            num1, den1 = Frac(frac1)
            num2, den2 = Frac(frac2)
        except TypeError:
            return NotImplemented
        new_den = math.lcm(den1, den2)
        new_num = num1*new_den//den1 + num2*new_den//den2
        return Frac((new_num, new_den))

    def __radd__(frac1: 'Frac', frac2: 'Frac'):
        return Frac.__add__(frac2, frac1)

    def __sub__(frac1: 'Frac', frac2: 'Frac'):
        try:
            return Frac.__add__(frac1, Frac.__neg__(frac2))
        except TypeError:
            return NotImplemented

    def __rsub__(frac1: 'Frac', frac2: 'Frac'):
        return Frac.__sub__(frac2, frac1)
    
    def __mul__(frac1: 'Frac', frac2: 'Frac'):
        try:
            num1, den1 = Frac(frac1)
            num2, den2 = Frac(frac2)
        except TypeError:
            return NotImplemented
        return Frac((num1 * num2, den1 * den2))
    
    def __rmul__(frac1: 'Frac', frac2: 'Frac'):
        return Frac.__mul__(frac2, frac1)

    def __truediv__(frac1: 'Frac', frac2: 'Frac'):
        try:
            return Frac.__mul__(frac1, Frac.inv(frac2))
        except TypeError:
            return NotImplemented
    
    def __rtruediv__(frac1: 'Frac', frac2: 'Frac'):
        return Frac.__truediv__(frac2, frac1)
    
    def __pow__(frac: 'Frac', power: int) -> 'Frac':
        if not isinstance(power, (int, np.signedinteger)):
            return NotImplemented
        if power < 0:
            return Frac(frac.den ** abs(power), frac.num ** abs(power))
        return Frac(frac.num ** power, frac.den ** power)
    
    def __eq__(frac1: 'Frac', frac2: 'Frac'):
        try:
            num1, den1 = Frac(frac1)
            num2, den2 = Frac(frac2)
        except TypeError:
            return NotImplemented
        return num1 == num2 and den1 == den2
    
    def __lt__(frac1: 'Frac', frac2: 'Frac'):
        try:
            num1, den1 = Frac(frac1)
            num2, den2 = Frac(frac2)
        except TypeError:
            return NotImplemented
        return num1 * den2 < num2 * den1

    def __le__(frac1: 'Frac', frac2: 'Frac'):
        try:
            num1, den1 = Frac(frac1)
            num2, den2 = Frac(frac2)
        except TypeError:
            return NotImplemented
        return num1 * den2 <= num2 * den1

    def __gt__(frac1: 'Frac', frac2: 'Frac'):
        try:
            num1, den1 = Frac(frac1)
            num2, den2 = Frac(frac2)
        except TypeError:
            return NotImplemented
        return num1 * den2 > num2 * den1

    def __ge__(frac1: 'Frac', frac2: 'Frac'):
        try:
            num1, den1 = Frac(frac1)
            num2, den2 = Frac(frac2)
        except AssertionError:
            return NotImplemented
        return num1 * den2 >= num2 * den1

    def __getitem__(self, index: int):
        if index == 0:
            return self.num
        elif index == 1:
            return self.den
        raise IndexError
        
    def __str__(self) -> str:
        return f"{self.num}/{self.den}"

    def __repr__(self) -> str:
        return self.__str__()

Fracs = lambda array: np.vectorize(Frac)(np.array(array, dtype=object))

def _main():
    assert Frac((1, 1)) == Frac(1) == Frac(1, 1)
    assert Frac(26, 13) == Frac(-26, -13) == (2, 1)
    assert Frac(-26, 13) == Frac(26, -13) == (-2, 1)
    assert -Frac(5, 8) + 2 * Frac(5, 3) - 1 - Frac(2, 3) / Frac(9, 7) / 5 == Frac(1733, 1080)
    assert Frac(5, 8) < Frac(2, 3)
    assert round(Frac(13, 2)) == 7
    assert Frac(13, 2).floordiv() == 6
    assert round(Frac(-26, 4)) == math.floor(-26/4 + 1/2)
    assert np.array_equal(Fracs([5, 6]) / 3, np.array([Frac(5, 3), Frac(2, 1)], dtype=object))
    assert np.dot(Fracs([5, 6]), Fracs([5, 6])) == Frac(61)
    assert np.dot(np.array([Frac(1, 2), Frac(2, 3)], dtype=object), np.array([Frac(3, 2), Frac(5, 2)], dtype=object)) == Frac(29, 12)
    assert np.array_equal(np.sum(Fracs([[1, 3], [4, 5]]), axis=1), Fracs([4, 9]))

if __name__ == "__main__":
    _main()