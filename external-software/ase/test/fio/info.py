from ase import Atoms
from ase.io import Trajectory

# Create a molecule with an info attribute
info = dict(creation_date='2011-06-27',
            chemical_name='Hydrogen',
            # custom classes also works provided that it is
            # imported and pickleable...
            foo={'seven': 7})

molecule = Atoms('H2', positions=[(0., 0., 0.), (0., 0., 1.1)], info=info)
assert molecule.info == info

# Copy molecule
atoms = molecule.copy()
assert atoms.info == info

# Save molecule to trajectory
traj = Trajectory('info.traj', 'w', atoms=molecule)
traj.write()
del traj

# Load molecule from trajectory
t = Trajectory('info.traj')
atoms = t[-1]

print(atoms.info)
assert atoms.info == info
