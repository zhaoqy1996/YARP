"""

Check the unit cell is handled correctly

"""

from ase.test.vasp import installed2 as installed
from ase.calculators.vasp import Vasp2 as Vasp
from ase.build import molecule
from ase.test import must_raise
assert installed()


# Molecules come with no unit cell

atoms = molecule('CH4')
calc = Vasp()

with must_raise(ValueError):
    atoms.set_calculator(calc)
    atoms.get_total_energy()
