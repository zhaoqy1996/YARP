from ase.calculators.lammpsrun import LAMMPS
from ase.cluster.icosahedron import Icosahedron
from ase.data import atomic_numbers,  atomic_masses
import numpy as np
from ase.optimize import LBFGS


ar_nc = Icosahedron('Ar', noshells=2)
ar_nc.cell = [[300, 0, 0], [0, 300, 0], [0, 0, 300]]
ar_nc.pbc = True

params = {}
params['pair_style'] = 'lj/cut 8.0'
params['pair_coeff'] = ['1 1 0.0108102 3.345']
params['mass'] = ['1 {}'.format(atomic_masses[atomic_numbers['Ar']])]

calc = LAMMPS(specorder=['Ar'], parameters=params)

ar_nc.set_calculator(calc)

E = ar_nc.get_potential_energy()
F = ar_nc.get_forces()

assert abs(E - -0.47) < 1E-2
assert abs(np.linalg.norm(F) - 0.0574) < 1E-4

dyn = LBFGS(ar_nc, force_consistent=False)
dyn.run(fmax=1E-6)

E = round(ar_nc.get_potential_energy(), 2)
F = ar_nc.get_forces()

assert abs(E - -0.48) < 1E-2
assert abs(np.linalg.norm(F) - 0.0) < 1E-5
