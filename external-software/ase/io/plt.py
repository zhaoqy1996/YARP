import numpy as np

from ase.atoms import Atoms
from ase.utils import basestring


def write_plt(filename, atoms, data):
    if isinstance(atoms, Atoms):
        cell = atoms.get_cell()
    else:
        cell = np.asarray(atoms, float)

    if cell.ndim == 2:
        c = cell.copy()
        cell = c.diagonal().copy()
        c.flat[::4] = 0.0
        if c.any():
            raise ValueError('Unit cell must be orthorhombic!')

    f = open(filename, 'w')
    np.array([3, 4], np.int32).tofile(f)

    dims = np.array(data.shape, np.int32)
    dims[::-1].tofile(f)

    for n, L in zip(dims[::-1], cell[::-1]):
        if n % 2 == 0:
            d = L / n
            np.array([0.0, L - d], np.float32).tofile(f)
        else:
            d = L / (n + 1)
            np.array([d, L - d], np.float32).tofile(f)

    if data.dtype == complex:
        data = np.abs(data)
    data.astype(np.float32).T.tofile(f)
    f.close()


def read_plt(fileobj):
    if isinstance(fileobj, basestring):
        fileobj = open(fileobj, 'rb')

    # dummy numbers
    np.fromfile(fileobj, dtype=np.int32, count=2)
    # read dimensions
    dims = np.fromfile(fileobj, dtype=np.int32, count=3)

    # read cell
    cell = np.zeros((3, 3), np.float32)
    for c in range(3):
        beg, Lmd = np.fromfile(fileobj, dtype=np.float32, count=2)
        n = dims[c]
        if n % 2 == 0:
            cell[2 - c, 2 - c] = Lmd / (1 - 1. / n)
        else:
            cell[2 - c, 2 - c] = Lmd / (1 - 1. / (n + 1))

    # read data
    data = np.fromfile(fileobj, dtype=np.float32)
    return data.reshape(dims).T, cell
