"""Test Atoms.__delitem__ with FixAtoms constraint."""
from ase import Atoms
from ase.constraints import FixBondLengths

a = Atoms('H3')
a.constraints = FixBondLengths([(1, 2)])
assert (a[:].constraints[0].pairs == [(1, 2)]).all()
assert (a[1:].constraints[0].pairs == [(0, 1)]).all()
assert len(a[2:].constraints) == 0
assert len(a[1:2].constraints) == 0
assert len(a[:2].constraints) == 0
assert len(a[:1].constraints) == 0

# Execise Atoms.__init__:
Atoms(a)
