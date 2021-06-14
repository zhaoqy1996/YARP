import numpy.random as random
import numpy as np
from ase import Atoms
from ase.neighborlist import (NeighborList, PrimitiveNeighborList,
                              NewPrimitiveNeighborList)
from ase.build import bulk

atoms = Atoms(numbers=range(10),
              cell=[(0.2, 1.2, 1.4),
                    (1.4, 0.1, 1.6),
                    (1.3, 2.0, -0.1)])
atoms.set_scaled_positions(3 * random.random((10, 3)) - 1)


def count(nl, atoms):
    c = np.zeros(len(atoms), int)
    R = atoms.get_positions()
    cell = atoms.get_cell()
    d = 0.0
    for a in range(len(atoms)):
        i, offsets = nl.get_neighbors(a)
        for j in i:
            c[j] += 1
        c[a] += len(i)
        d += (((R[i] + np.dot(offsets, cell) - R[a])**2).sum(1)**0.5).sum()
    return d, c

for sorted in [False, True]:
    for p1 in range(2):
        for p2 in range(2):
            for p3 in range(2):
                # print(p1, p2, p3)
                atoms.set_pbc((p1, p2, p3))
                nl = NeighborList(atoms.numbers * 0.2 + 0.5,
                                  skin=0.0, sorted=sorted)
                nl.update(atoms)
                d, c = count(nl, atoms)
                atoms2 = atoms.repeat((p1 + 1, p2 + 1, p3 + 1))
                nl2 = NeighborList(atoms2.numbers * 0.2 + 0.5,
                                   skin=0.0, sorted=sorted)
                nl2.update(atoms2)
                d2, c2 = count(nl2, atoms2)
                c2.shape = (-1, 10)
                dd = d * (p1 + 1) * (p2 + 1) * (p3 + 1) - d2
                assert abs(dd) < 1e-10
                assert not (c2 - c).any()

h2 = Atoms('H2', positions=[(0, 0, 0), (0, 0, 1)])
nl = NeighborList([0.5, 0.5], skin=0.1, sorted=True, self_interaction=False)
nl2 = NeighborList([0.5, 0.5], skin=0.1, sorted=True, self_interaction=False, primitive=NewPrimitiveNeighborList)
assert nl2.update(h2)
assert nl.update(h2)
assert not nl.update(h2)
assert (nl.get_neighbors(0)[0] == [1]).all()
m = np.zeros((2,2))
m[0,1] = 1
assert np.array_equal(nl.get_connectivity_matrix(sparse=False), m)
assert np.array_equal(nl.get_connectivity_matrix(sparse=True).todense(), m)
assert np.array_equal(nl.get_connectivity_matrix().todense(), nl2.get_connectivity_matrix().todense())

h2[1].z += 0.09
assert not nl.update(h2)
assert (nl.get_neighbors(0)[0] == [1]).all()

h2[1].z += 0.09
assert nl.update(h2)
assert (nl.get_neighbors(0)[0] == []).all()
assert nl.nupdates == 2

h2 = Atoms('H2', positions=[(0, 0, 0), (0, 0, 1)])
nl = NeighborList([0.1, 0.1], skin=0.1, bothways=True, self_interaction=False)
assert nl.update(h2)
assert nl.get_neighbors(0)[1].shape == (0, 3)
assert nl.get_neighbors(0)[1].dtype == int

x = bulk('X', 'fcc', a=2**0.5)

nl = NeighborList([0.5], skin=0.01, bothways=True, self_interaction=False)
nl.update(x)
assert len(nl.get_neighbors(0)[0]) == 12

nl = NeighborList([0.5] * 27, skin=0.01, bothways=True, self_interaction=False)
nl.update(x * (3, 3, 3))
for a in range(27):
    assert len(nl.get_neighbors(a)[0]) == 12
assert not np.any(nl.get_neighbors(13)[1])

c = 0.0058
for NeighborListClass in [PrimitiveNeighborList, NewPrimitiveNeighborList]:
    nl = NeighborListClass([c, c],
                           skin=0.0,
                           sorted=True,
                           self_interaction=False,
                           use_scaled_positions=True)
    nl.update([True, True, True],
              np.eye(3) * 7.56,
              np.array([[0, 0, 0],
                        [0, 0, 0.99875]]))
    n0, d0 = nl.get_neighbors(0)
    n1, d1 = nl.get_neighbors(1)
    # != is xor
    assert (np.all(n0 == [0]) and np.all(d0 == [0, 0, 1])) != \
        (np.all(n1 == [1]) and np.all(d1 == [0, 0, -1]))

# Test empty neighbor list
nl = PrimitiveNeighborList([])
nl.update([True, True, True],
          np.eye(3) * 7.56,
          np.zeros((0, 3)))

# Test hexagonal cell and large cutoff
pbc_c = np.array([True, True, True])
cutoff_a = np.array([8.0, 8.0])
cell_cv = np.array([[0., 3.37316113, 3.37316113],
                    [3.37316113, 0., 3.37316113],
                    [3.37316113, 3.37316113, 0.]])
spos_ac = np.array([[0., 0., 0.],
                    [0.25, 0.25, 0.25]])

nl = PrimitiveNeighborList(cutoff_a, skin=0.0, sorted=True, use_scaled_positions=True)
nl2 = NewPrimitiveNeighborList(cutoff_a, skin=0.0, sorted=True, use_scaled_positions=True)
nl.update(pbc_c, cell_cv, spos_ac)
nl2.update(pbc_c, cell_cv, spos_ac)

a0, offsets0 = nl.get_neighbors(0)
b0 = np.zeros_like(a0)
d0 = np.dot(spos_ac[a0] + offsets0 - spos_ac[0], cell_cv)
a1, offsets1 = nl.get_neighbors(1)
d1 = np.dot(spos_ac[a1] + offsets1 - spos_ac[1], cell_cv)
b1 = np.ones_like(a1)

a = np.concatenate([a0, a1])
b = np.concatenate([b0, b1])
d = np.concatenate([d0, d1])
_a = np.concatenate([a, b])
_b = np.concatenate([b, a])
a = _a
b = _b
d = np.concatenate([d, -d])

a0, offsets0 = nl2.get_neighbors(0)
d0 = np.dot(spos_ac[a0] + offsets0 - spos_ac[0], cell_cv)
b0 = np.zeros_like(a0)
a1, offsets1 = nl2.get_neighbors(1)
d1 = np.dot(spos_ac[a1] + offsets1 - spos_ac[1], cell_cv)
b1 = np.ones_like(a1)

a2 = np.concatenate([a0, a1])
b2 = np.concatenate([b0, b1])
d2 = np.concatenate([d0, d1])
_a2 = np.concatenate([a2, b2])
_b2 = np.concatenate([b2, a2])
a2 = _a2
b2 = _b2
d2 = np.concatenate([d2, -d2])

i = np.argsort(d[:, 0]+d[:, 1]*1e2+d[:, 2]*1e4+a*1e6)
i2 = np.argsort(d2[:, 0]+d2[:, 1]*1e2+d2[:, 2]*1e4+a2*1e6)

assert np.all(a[i] == a2[i2])
assert np.all(b[i] == b2[i2])
assert np.allclose(d[i], d2[i2])
