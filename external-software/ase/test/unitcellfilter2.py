import numpy as np

from ase.build import bulk
from ase.calculators.test import gradient_test
from ase.calculators.lj import LennardJones
from ase.constraints import UnitCellFilter, ExpCellFilter

a0 = bulk('Cu', cubic=True)

# perturb the atoms
s = a0.get_scaled_positions()
s[:, 0] *= 0.995
a0.set_scaled_positions(s)

# perturb the cell
a0.cell[...] += np.random.uniform(-1e-2, 1e-2,
                                  size=9).reshape((3,3))

atoms = a0.copy()
atoms.set_calculator(LennardJones())
ucf = UnitCellFilter(atoms)

# test all deritatives
f, fn = gradient_test(ucf)
assert abs(f - fn).max() < 1e-6

atoms = a0.copy()
atoms.set_calculator(LennardJones())
ecf = ExpCellFilter(atoms)

# test all deritatives
f, fn = gradient_test(ecf)
assert abs(f - fn).max() < 1e-6
