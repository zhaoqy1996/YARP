from ase.calculators.lammpsrun import LAMMPS
from numpy.linalg import norm
from ase.test.eam_pot import Pt_u3
from ase.build import fcc111
import os


pot_fn = 'Pt_u3.eam'
f = open(pot_fn, 'w')
f.write(Pt_u3)
f.close()

slab = fcc111('Pt', size=(10, 10, 5), vacuum=30.0)

params = {}
params['pair_style'] = 'eam'
params['pair_coeff'] = ['1 1 {}'.format(pot_fn)]

calc = LAMMPS(specorder=['Pt'], parameters=params, files=[pot_fn])
slab.set_calculator(calc)
E = slab.get_potential_energy()
F = slab.get_forces()

assert abs(E - -2758.63) < 1E-2
assert abs(norm(F) - 11.3167) < 1E-4
assert abs(norm(slab.positions) - 955.259) < 1E-3

params['group'] = ['lower_atoms id '
                   + ' '.join([str(i+1) for i,
                              tag in enumerate(slab.get_tags()) if tag >= 4])]
params['fix'] = ['freeze_lower_atoms lower_atoms setforce 0.0 0.0 0.0']
params['run'] = 100
params['timestep'] = 0.0005
calc.parameters = params
calc.write_velocities = True
calc.dump_period = 10
# set_atoms=True to read final coordinates and velocities after NVE simulation
calc.run(set_atoms=True)

new_slab = calc.atoms.copy()

Ek = new_slab.get_kinetic_energy()
Ek2 = calc.thermo_content[-1]['ke']
# do not use  slab.get_potential_energy()
# because it will run NVE simulation again
E = calc.thermo_content[-1]['pe']
T = calc.thermo_content[-1]['temp']

assert abs(Ek - Ek2) < 1E-4
assert abs(Ek - 2.53) < 1E-2
assert abs(E - -2761.17) < 1E-2
assert abs(norm(new_slab.positions) - 871.993) < 1E-3

os.remove(pot_fn)
