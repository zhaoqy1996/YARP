"""Test that FixAtoms constraints are written properly."""
from ase.build import molecule
from ase.constraints import FixAtoms
from ase.io import write, read

mol1 = molecule('H2O')
mol1.set_constraint(FixAtoms(mask=[True, False, True]))
write('coord', mol1)
mol2 = read('coord')
fix_indices1 = mol1.constraints[0].get_indices()
fix_indices2 = mol2.constraints[0].get_indices()
assert all(fix_indices1 == fix_indices2)
