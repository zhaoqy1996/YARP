from ase.calculators.lammpsrun import LAMMPS
from ase.cluster.icosahedron import Icosahedron
from ase.data import atomic_numbers,  atomic_masses
from numpy.linalg import norm

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
assert abs(norm(F) - 0.0574) < 1E-4
assert abs(norm(ar_nc.positions) - 23.588) < 1E-3


params['minimize'] = '1.0e-15 1.0e-6 2000 4000'   # add minimize
calc.params = params

# set_atoms=True to read final coordinates after minimization
calc.run(set_atoms=True)

# get final coordinates after minimization
ar_nc.set_positions(calc.atoms.positions)

E = ar_nc.get_potential_energy()
F = ar_nc.get_forces()

assert abs(E - -0.48) < 1E-2
assert abs(norm(F) - 0.0) < 1E-6
assert abs(norm(ar_nc.positions) - 23.399) < 1E-3
