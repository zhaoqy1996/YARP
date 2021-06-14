from __future__ import division

import numpy as np

import ase
import ase.lattice.hexagonal
from ase.build import bulk, molecule

from ase.neighborlist import (mic, neighbor_list, primitive_neighbor_list,
                              first_neighbors)

tol = 1e-7

# two atoms
a = ase.Atoms('CC', positions=[[0.5, 0.5, 0.5], [1,1,1]], cell=[10, 10, 10],
              pbc=True)
i, j, d = neighbor_list("ijd", a, 1.1)
assert (i == np.array([0, 1])).all()
assert (j == np.array([1, 0])).all()
assert np.abs(d - np.array([np.sqrt(3/4), np.sqrt(3/4)])).max() < tol

# test_neighbor_list
for pbc in [True, False, [True, False, True]]:
    a = ase.Atoms('4001C', cell=[29, 29, 29])
    a.set_scaled_positions(np.transpose([np.random.random(len(a)),
                                         np.random.random(len(a)),
                                         np.random.random(len(a))]))
    j, dr, i, abs_dr, shift = neighbor_list("jDidS", a, 1.85)

    assert (np.bincount(i) == np.bincount(j)).all()

    r = a.get_positions()
    dr_direct = mic(r[j]-r[i], a.cell)
    assert np.abs(r[j]-r[i]+shift.dot(a.cell) - dr_direct).max() < tol

    abs_dr_from_dr = np.sqrt(np.sum(dr*dr, axis=1))
    abs_dr_direct = np.sqrt(np.sum(dr_direct*dr_direct, axis=1))

    assert np.all(np.abs(abs_dr-abs_dr_from_dr) < 1e-12)
    assert np.all(np.abs(abs_dr-abs_dr_direct) < 1e-12)

    assert np.all(np.abs(dr-dr_direct) < 1e-12)

# test_neighbor_list_atoms_outside_box
for pbc in [True, False, [True, False, True]]:
    a = ase.Atoms('4001C', cell=[29, 29, 29])
    a.set_scaled_positions(np.transpose([np.random.random(len(a)),
                                         np.random.random(len(a)),
                                         np.random.random(len(a))]))
    a.set_pbc(pbc)
    a.positions[100, :] += a.cell[0, :]
    a.positions[200, :] += a.cell[1, :]
    a.positions[300, :] += a.cell[2, :]
    j, dr, i, abs_dr, shift = neighbor_list("jDidS", a, 1.85)

    assert (np.bincount(i) == np.bincount(j)).all()

    r = a.get_positions()
    dr_direct = mic(r[j]-r[i], a.cell)
    assert np.abs(r[j]-r[i]+shift.dot(a.cell) - dr_direct).max() < tol

    abs_dr_from_dr = np.sqrt(np.sum(dr*dr, axis=1))
    abs_dr_direct = np.sqrt(np.sum(dr_direct*dr_direct, axis=1))

    assert np.all(np.abs(abs_dr-abs_dr_from_dr) < 1e-12)
    assert np.all(np.abs(abs_dr-abs_dr_direct) < 1e-12)

    assert np.all(np.abs(dr-dr_direct) < 1e-12)

# test_small_cell
a = ase.Atoms('C', positions=[[0.5, 0.5, 0.5]], cell=[1, 1, 1],
              pbc=True)
i, j, dr, shift = neighbor_list("ijDS", a, 1.1)
assert np.bincount(i)[0] == 6
assert (dr == shift).all()

i, j = neighbor_list("ij", a, 1.5)
assert np.bincount(i)[0] == 18

a.set_pbc(False)
i = neighbor_list("i", a, 1.1)
assert len(i) == 0

a.set_pbc([True, False, False])
i = neighbor_list("i", a, 1.1)
assert np.bincount(i)[0] == 2

a.set_pbc([True, False, True])
i = neighbor_list("i", a, 1.1)
assert np.bincount(i)[0] == 4

# test_out_of_cell_small_cell
a = ase.Atoms('CC', positions=[[0.5, 0.5, 0.5],
                               [1.1, 0.5, 0.5]],
              cell=[1, 1, 1], pbc=False)
i1, j1, r1 = neighbor_list("ijd", a, 1.1)
a.set_cell([2, 1, 1])
i2, j2, r2 = neighbor_list("ijd", a, 1.1)

assert (i1 == i2).all()
assert (j1 == j2).all()
assert np.abs(r1 - r2).max() < tol

# test_out_of_cell_large_cell
a = ase.Atoms('CC', positions=[[9.5, 0.5, 0.5],
                               [10.1, 0.5, 0.5]],
              cell=[10, 10, 10], pbc=False)
i1, j1, r1 = neighbor_list("ijd", a, 1.1)
a.set_cell([20, 10, 10])
i2, j2, r2 = neighbor_list("ijd", a, 1.1)

assert (i1 == i2).all()
assert (j1 == j2).all()
assert np.abs(r1 - r2).max() < tol

# test_hexagonal_cell
for sx in range(3):
    a = ase.lattice.hexagonal.Graphite('C', latticeconstant=(2.5, 10.0),
                                       size=[sx+1,sx+1,1])
    i = neighbor_list("i", a, 1.85)
    assert np.all(np.bincount(i)==3)

# test_first_neighbors
i = [1,1,1,1,3,3,3]
assert (first_neighbors(5, i) == np.array([0,0,4,4,7,7])).all()
i = [0,1,2,3,4,5]
assert (first_neighbors(6, i) == np.array([0,1,2,3,4,5,6])).all()

# test_multiple_elements
a = molecule('HCOOH')
a.center(vacuum=5.0)
i = neighbor_list("i", a, 1.85)
assert (np.bincount(i) == np.array([2,3,1,1,1])).all()

cutoffs = {(1, 6): 1.2}
i = neighbor_list("i", a, cutoffs)
assert (np.bincount(i) == np.array([0,1,0,0,1])).all()

cutoffs = {(6, 8): 1.4}
i = neighbor_list("i", a, cutoffs)
assert (np.bincount(i) == np.array([1,2,1])).all()

cutoffs = {('H', 'C'): 1.2, (6, 8): 1.4}
i = neighbor_list("i", a, cutoffs)
assert (np.bincount(i) == np.array([1,3,1,0,1])).all()

cutoffs = [0.0, 0.9, 0.0, 0.5, 0.5]
i = neighbor_list("i", a, cutoffs)
assert (np.bincount(i) == np.array([0,1,0,0,1])).all()

cutoffs = [0.7, 0.9, 0.7, 0.5, 0.5]
i = neighbor_list("i", a, cutoffs)
assert (np.bincount(i) == np.array([2,3,1,1,1])).all()

# test_noncubic
a = bulk("Al", cubic=False)
i, j, d = neighbor_list("ijd", a, 3.1)
assert (np.bincount(i) == np.array([12])).all()
assert np.abs(d - [2.86378246]*12).max() < tol

# test pbc
nat = 10
atoms = ase.Atoms(numbers=range(nat),
                  cell=[(0.2, 1.2, 1.4),
                        (1.4, 0.1, 1.6),
                        (1.3, 2.0, -0.1)])
atoms.set_scaled_positions(3 * np.random.random((nat, 3)) - 1)

for p1 in range(2):
    for p2 in range(2):
        for p3 in range(2):
            atoms.set_pbc((p1, p2, p3))
            i, j, d, D, S = neighbor_list("ijdDS", atoms, atoms.numbers * 0.2 + 0.5)
            c = np.bincount(i, minlength=len(atoms))
            atoms2 = atoms.repeat((p1 + 1, p2 + 1, p3 + 1))
            i2, j2, d2, D2, S2 = neighbor_list("ijdDS", atoms2, atoms2.numbers * 0.2 + 0.5)
            c2 = np.bincount(i2, minlength=len(atoms))
            c2.shape = (-1, nat)
            dd = d.sum() * (p1 + 1) * (p2 + 1) * (p3 + 1) - d2.sum()
            dr = np.linalg.solve(atoms.cell.T, (atoms.positions[1]-atoms.positions[0]).T).T+np.array([0,0,3])
            assert abs(dd) < 1e-10
            assert not (c2 - c).any()

c = 0.0058
i, j, d = primitive_neighbor_list('ijd',
                                  [True, True, True],
                                  np.eye(3) * 7.56,
                                  np.array([[0, 0, 0],
                                            [0, 0, 0.99875]]),
                                  [c, c],
                                  self_interaction=False,
                                  use_scaled_positions=True)
assert np.all(i == [0, 1])
assert np.all(j == [1, 0])
assert np.allclose(d, [0.00945, 0.00945])

# Empty atoms object
i, D, d, j, S = neighbor_list("iDdjS", ase.Atoms(), 1.0)
assert i.dtype == np.int
assert j.dtype == np.int
assert d.dtype == np.float
assert D.dtype == np.float
assert S.dtype == np.int
assert i.shape == (0,)
assert j.shape == (0,)
assert d.shape == (0,)
assert D.shape == (0, 3)
assert S.shape == (0, 3)

# Check that only a scalar (not a tuple) is returned if we request a single
# argument.
i = neighbor_list("i", ase.Atoms(), 1.0)
assert i.dtype == np.int
assert i.shape == (0,)
