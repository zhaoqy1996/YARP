"""Checks that writing and reading of NWChem input files is consistent."""

from ase.build import molecule
from ase.calculators.nwchem import NWChem
from ase import io

atoms = molecule('CH3COOH')

calc = NWChem()
calc.write_input(atoms)

atoms2 = io.read('nwchem.nw')

tol = 1e-8

check = sum(abs((atoms.positions - atoms2.positions).ravel()) > tol)
assert check == 0
