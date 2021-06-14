"""Test that set_momenta behaves as expected when constraints are
involved."""

import numpy as np
from ase import Atoms, Atom
from ase.constraints import Hookean, FixAtoms


# FixAtoms check
atoms = Atoms([Atom('H', (0., 0., 0.)),
               Atom('H', (2., 0., 0.))])
atoms.set_constraint(FixAtoms(indices=[0]))
atoms.set_momenta(np.ones(atoms.get_momenta().shape))
desired = np.ones(atoms.get_momenta().shape)
desired[0] = 0.
actual = atoms.get_momenta()
assert (actual == desired).all()

# Hookean check
atoms = Atoms([Atom('H', (0., 0., 0.)),
               Atom('H', (2., 0., 0.))])
atoms.set_constraint(Hookean(0, 1, rt=1., k=10.))
atoms.set_momenta(np.zeros(atoms.get_momenta().shape))
actual = atoms.get_momenta()
desired = np.zeros(atoms.get_momenta().shape)
# Disabled for now:
# assert (actual == desired).all()
