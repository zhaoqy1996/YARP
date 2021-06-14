import numpy as np
from ase import Atoms

charges = np.array([-1, 1])
a = Atoms('H2', positions=[(0, 0, 0), (0, 0, 1.1)], charges=charges)

a.pbc[0] = 1
assert a.pbc.any()
assert not a.pbc.all()
a.pbc = 1
assert a.pbc.all()

a.cell = (1, 2, 3)
a.cell *= 2
a.cell[0, 0] = 3
assert not (a.cell.diagonal() - (3, 4, 6)).any()

assert (charges == a.get_initial_charges()).all()
assert a.has('initial_charges')
# XXX extend has to calculator properties
assert not a.has('charges')
