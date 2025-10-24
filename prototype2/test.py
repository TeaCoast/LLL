from fracs import Fracs
import numpy as np

fracs = Fracs(np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]))

hello = np.array([[1, 2, 3]])
print(fracs.nums * hello.T)