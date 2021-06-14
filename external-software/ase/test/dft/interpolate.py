import numpy as np
from ase.dft.kpoints import monkhorst_pack_interpolate

eps = [0, 1, 2]
path = [[0, 0, 0], [-0.25, 0, 0]]
bz2ibz = [0, 1, 1, 2]
x = monkhorst_pack_interpolate(path, eps, np.eye(3), bz2ibz,
                               [2, 2, 1], [0.25, 0.25, 0])
print(x)
assert abs(x - [0, 0.5]).max() < 1e-10

