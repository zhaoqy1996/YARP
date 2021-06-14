"""This test makes sure that the forces returned from a
SinglePointCalculator are immutable. Previously, successive calls to
atoms.get_forces(apply_constraint=x), with x alternating between True and
False, would get locked into the constrained variation."""

from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.io import read
from ase.constraints import FixAtoms


def check_forces():
    """Makes sure the unconstrained forces stay that way."""
    forces = atoms.get_forces(apply_constraint=False)
    funconstrained = float(forces[0, 0])

    forces = atoms.get_forces(apply_constraint=True)

    forces = atoms.get_forces(apply_constraint=False)
    funconstrained2 = float(forces[0, 0])

    assert funconstrained2 == funconstrained

atoms = fcc111('Cu', (2, 2, 1), vacuum=10.)
atoms[0].x += 0.2
atoms.set_constraint(FixAtoms(indices=[atom.index for atom in atoms]))

# First run the tes with EMT and save a force component.
atoms.set_calculator(EMT())
check_forces()
f = float(atoms.get_forces(apply_constraint=False)[0, 0])

# Save and reload with a SinglePointCalculator.
atoms.write('singlepointtest.traj')
atoms = read('singlepointtest.traj')
check_forces()

# Manually change a value.
forces = atoms.get_forces(apply_constraint=False)
forces[0, 0] = 42.
forces = atoms.get_forces(apply_constraint=False)
assert forces[0, 0] == f
