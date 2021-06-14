import numpy as np

from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.optimize.precon import Exp, PreconLBFGS

cu0 = bulk("Cu") * (2, 2, 2)
sigma = cu0.get_distance(0,1)*(2.**(-1./6))
lj = LennardJones(sigma=sigma)

# perturb the cell
cell = cu0.get_cell()
cell *= 0.95
cell[1,0] += 0.2
cell[2,1] += 0.5
cu0.set_cell(cell, scale_atoms=True)

energies = []
for use_armijo in [True, False]:
    for a_min in [None, 1e-3]:
        atoms = cu0.copy()
        atoms.set_calculator(lj)
        opt = PreconLBFGS(atoms, precon=Exp(A=3), use_armijo=use_armijo,
                          a_min=a_min, variable_cell=True)
        opt.run(fmax=1e-3, smax=1e-4)
        energies.append(atoms.get_potential_energy())

# check we get the expected energy for all methods
assert np.abs(np.array(energies) - -63.5032311942).max() < 1e-4
