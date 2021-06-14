"""
Test of Hookean constraint.

Checks for activity in keeping a bond, preventing vaporization, and
that energy is conserved in NVE dynamics.
"""

import numpy as np
from ase import Atoms, Atom
from ase.build import fcc110
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, Hookean
from ase.md import VelocityVerlet
from ase import units


class SaveEnergy:
    """Class to save energy."""

    def __init__(self, atoms):
        self.atoms = atoms
        self.energies = []

    def __call__(self):
        self.energies.append(atoms.get_total_energy())


# Make Pt 110 slab with Cu2 adsorbate.
atoms = fcc110('Pt', (2, 2, 2), vacuum=7.)
adsorbate = Atoms([Atom('Cu', atoms[7].position + (0., 0., 2.5)),
                   Atom('Cu', atoms[7].position + (0., 0., 5.0))])
atoms.extend(adsorbate)
calc = EMT()
atoms.set_calculator(calc)

# Constrain the surface to be fixed and a Hookean constraint between
# the adsorbate atoms.
constraints = [FixAtoms(indices=[atom.index for atom in atoms if
                                 atom.symbol == 'Pt']),
               Hookean(a1=8, a2=9, rt=2.6, k=15.),
               Hookean(a1=8, a2=(0., 0., 1., -15.), k=15.)]
atoms.set_constraint(constraints)

# Give it some kinetic energy.
momenta = atoms.get_momenta()
momenta[9, 2] += 20.
momenta[9, 1] += 2.
atoms.set_momenta(momenta)

# Propagate in Velocity Verlet (NVE).
dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
energies = SaveEnergy(atoms)
dyn.attach(energies)
dyn.run(steps=100)

# Test the max bond length and position.
bondlength = np.linalg.norm(atoms[8].position - atoms[9].position)
assert bondlength < 3.0
assert atoms[9].z < 15.0

# Test that energy was conserved.
assert max(energies.energies) - min(energies.energies) < 0.01

# Make sure that index shuffle works.
neworder = list(range(len(atoms)))
neworder[8] = 9  # Swap two atoms.
neworder[9] = 8
atoms = atoms[neworder]
assert atoms.constraints[1].indices[0] == 9
assert atoms.constraints[1].indices[1] == 8
assert atoms.constraints[2].index == 9
