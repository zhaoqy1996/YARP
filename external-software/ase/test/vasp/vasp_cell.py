"""

Check the unit cell is handled correctly

"""

from ase.calculators.vasp import Vasp
from ase.build import molecule
from ase.test import must_raise

# Molecules come with no unit cell

atoms = molecule('CH4')
calc = Vasp()

with must_raise(RuntimeError):
    atoms.write('POSCAR')

with must_raise(ValueError):
    atoms.set_calculator(calc)
    atoms.get_total_energy()
