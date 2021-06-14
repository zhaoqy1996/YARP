from ase.build import molecule
from ase.constraints import FixAtoms

N = 2

atoms = molecule('CO2')
atoms.set_cell((15, 15, 15))

# Indices method:
atomsi = atoms.copy()
atomsi.set_constraint(FixAtoms(indices=[0]))
atomsi = atomsi.repeat((N, 1, 1))

atomsiref = atoms.copy().repeat((N, 1, 1))
atomsiref.set_constraint(FixAtoms(indices=list(range(0, 3 * N, 3))))

lcatomsi = list(atomsi.constraints[0].index)
lcatomsiref = list(atomsiref.constraints[0].index)

assert lcatomsi == lcatomsiref

# Mask method:
atomsm = atoms.copy()
atomsm.set_constraint(FixAtoms(mask=[True, False, False]))
atomsm = atomsm.repeat((N, 1, 1))

atomsmref = atoms.copy().repeat((N, 1, 1))
atomsmref.set_constraint(FixAtoms(mask=[True, False, False] * N))

lcatomsm = list(atomsm.constraints[0].index)
lcatomsmref = list(atomsmref.constraints[0].index)

assert lcatomsm == lcatomsmref
assert lcatomsm == lcatomsi
