from ase.build import bulk
from ase.calculators.dmol import DMol3

atoms = bulk('Al')
calc = DMol3()
atoms.set_calculator(calc)
atoms.get_potential_energy()
atoms.get_forces()
