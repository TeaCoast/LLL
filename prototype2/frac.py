import math
import numpy as np

def _simplify(frac: tuple[int, int]):
    num, den = frac
    num *= int(np.sign(den))
    den = abs(den)
    gcd = math.gcd(num, den)
    return (num // gcd, den // gcd)

class Frac(tuple[int, int]):
    def __new__(self, frac: tuple[int, int] | int, den: int = 1):
        if isinstance(frac, int) and isinstance(den, int):
            frac = (frac, den)
        elif not isinstance(frac, tuple) or not len(frac) == 2 or den != 1:
            raise TypeError
        elif not isinstance(frac[0], (int, np.signedinteger)) or not isinstance(frac[1], (int, np.signedinteger)):
            raise TypeError
        if frac[1] == 0:
            raise ZeroDivisionError
        return super().__new__(self, _simplify(frac))

    def inv(frac: 'Frac') -> 'Frac':
        try:
            num, den = Frac(frac)
        except TypeError:
            return NotImplemented
        return Frac((den*np.sign(num), abs(num)))
    
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
            return tuple(Frac(frac1)) == tuple(Frac(frac2))
        except TypeError:
            return NotImplemented
    
    def __ne__(frac1: 'Frac', frac2: 'Frac'):
        try:
            return tuple(Frac(frac1)) != tuple(Frac(frac2))
        except TypeError:
            return NotImplemented
    
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
        except TypeError:
            return NotImplemented
        return num1 * den2 >= num2 * den1
    
    @property
    def num(self) -> int:
        return self[0]
    
    @property
    def den(self) -> int:
        return self[1]

    def __str__(frac: 'Frac') -> str:
        try:
            num, den = Frac(frac)
        except TypeError:
            return NotImplemented
        return f"{num}/{den}"

def _main():
    assert Frac((1, 1)) == Frac(1) == Frac(1, 1)
    assert tuple(Frac(26, 13)) == tuple(Frac(-26, -13)) == (2, 1)
    assert tuple(Frac(-26, 13)) == tuple(Frac(26, -13)) == (-2, 1)
    assert -Frac(5, 8) + 2 * Frac(5, 3) - 1 - Frac(2, 3) / Frac(9, 7) / 5 == Frac(1733, 1080)
    assert Frac(5, 8) < Frac(2, 3)
    assert Frac(13, 2).round() == 7
    assert Frac(13, 2).floordiv() == 6
    print(Frac(-26, 4).round(), Frac(-26, 4).floordiv())
    print(round(-5.5), round(-4.5))

if __name__ == "__main__":
    _main()