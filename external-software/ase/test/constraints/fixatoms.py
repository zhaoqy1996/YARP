"""Test Atoms.__delitem__ with FixAtoms constraint."""
from ase import Atoms
from ase.constraints import FixAtoms

for i, j in [(slice(0, -1), None),
             (slice(0, 1), [0]),
             (slice(0, None), None),
             (0, [0]),
             (1, [0]),
             (2, [0, 1]),
             (-1, [0, 1])]:
    a = Atoms('H3')
    a.constraints = FixAtoms(indices=[0, 1])
    del a[i]
    print(i, j, a.constraints)
    if j is None:
        assert len(a.constraints) == 0
    else:
        assert (a.constraints[0].index == j).all()
