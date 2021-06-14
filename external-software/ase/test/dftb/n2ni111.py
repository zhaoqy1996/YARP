"""Test run for DFTB+ calculator.

Use these .skf-files from dftb.org:

    * trans3d/trans3d-0-1/Ni-Ni.skf
    * trans3d/trans3d-0-1/Ni-N.skf
    * mio/mio-1-1/N-N.skf

"""
import os

from ase import Atoms
from ase.calculators.dftb import Dftb
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.build import fcc111, add_adsorbate

h = 1.85
d = 1.10
k = 2

slab = fcc111('Ni', size=(2, 2, 3), vacuum=10.0)
calc1 = Dftb(label='slab',
             kpts=[k, k, 1],
             Hamiltonian_SCC='YES')
slab.set_calculator(calc1)
dyn = QuasiNewton(slab, trajectory='slab.traj')
dyn.run(fmax=0.05)
e_slab = slab.get_potential_energy()
os.system('rm dftb_in.hsd')
molecule = Atoms('2N', positions=[(0., 0., 0.), (0., 0., d)])
calc2 = Dftb(label='n2',
             Hamiltonian_SCC='YES')
molecule.set_calculator(calc2)
dyn = QuasiNewton(molecule, trajectory='n2.traj')
dyn.run(fmax=0.05)
e_N2 = molecule.get_potential_energy()

slab2 = slab
add_adsorbate(slab2, molecule, h, 'ontop')
constraint = FixAtoms(mask=[a.symbol != 'N' for a in slab2])
slab2.set_constraint(constraint)
calc3 = Dftb(label='slab2',
             kpts=[k, k, 1],
             Hamiltonian_SCC='YES')
slab2.set_calculator(calc3)
dyn = QuasiNewton(slab2, trajectory='slab2.traj')
dyn.run(fmax=0.05)

adsorption_energy = e_slab + e_N2 - slab2.get_potential_energy()
print(adsorption_energy, 'eV')
assert abs(adsorption_energy - -0.30) < 0.1
