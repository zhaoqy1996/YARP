import os

from ase import Atoms, Atom
from ase.io import write
from ase.calculators.jacapo import Jacapo

atoms = Atoms([Atom('H',[0,0,0])],
              cell=(2,2,2),
              pbc=True)

calc = Jacapo('Jacapo-test.nc',
              pw=200,
              nbands=2,
              kpts=(1,1,1),
              spinpol=False,
              dipole=False,
              symmetry=False,
              ft=0.01)

atoms.set_calculator(calc)

print(atoms.get_potential_energy())
write('Jacapo-test.traj', atoms)
os.system('rm -f Jacapo-test.nc Jacapo-test.txt Jacapo-test.traj')
