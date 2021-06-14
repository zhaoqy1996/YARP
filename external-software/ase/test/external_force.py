"""Tests for class ExternalForce in ase/constraints.py"""
from ase import Atoms
from ase.constraints import ExternalForce, FixBondLength
from ase.optimize import FIRE
from ase.calculators.emt import EMT
from numpy.linalg import norm


f_ext = 0.2

atom1 = 0
atom2 = 1

atom3 = 2

fmax = 0.001

atoms = Atoms('H3', positions=[(0, 0, 0), (0.751, 0, 0), (0, 1., 0)])
atoms.set_calculator(EMT())

# Without external force
opt = FIRE(atoms)
opt.run(fmax=fmax)
dist1 = atoms.get_distance(atom1, atom2)

# With external force
con1 = ExternalForce(atom1, atom2, f_ext)
atoms.set_constraint(con1)
opt = FIRE(atoms)
opt.run(fmax=fmax)
dist2 = atoms.get_distance(atom1, atom2)
# Distance should increase due to the external force
assert dist2 > dist1

# Combine ExternalForce with FixBondLength

# Fix the bond on which the force acts
con2 = FixBondLength(atom1, atom2)
# ExternalForce constraint at the beginning of the list!!!
atoms.set_constraint([con1, con2])
opt = FIRE(atoms)
opt.run(fmax=fmax)
f_con = con2.constraint_forces

# It was already optimized with this external force, therefore
# the constraint force should be almost zero
assert norm(f_con[0]) <= fmax

# To get the complete constraint force (with external force),
# use only the FixBondLength constraint, after the optimization with
# ExternalForce
atoms.set_constraint(con2)
opt = FIRE(atoms)
opt.run(fmax=fmax)
f_con = con2.constraint_forces[0]
assert round(norm(f_con), 2) == round(abs(f_ext), 2)

# Fix another bond and incrase the external force
f_ext *= 2
con1 = ExternalForce(atom1, atom2, f_ext)
d1 = atoms.get_distance(atom1, atom3)
con2 = FixBondLength(atom1, atom3)
# ExternalForce constraint at the beginning of the list!!!
atoms.set_constraint([con1, con2])
opt = FIRE(atoms)
opt.run(fmax=fmax)
d2 = atoms.get_distance(atom1, atom3)
# Fixed distance should not change
assert round(d1, 5) == round(d2, 5)
