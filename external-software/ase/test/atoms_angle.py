from ase import Atoms
import numpy as np

atoms = Atoms(['O', 'H', 'H'], positions=[[0., 0., 0.119262],
                                          [0., 0.763239, -0.477047],
                                          [0., -0.763239, -0.477047]])

# Angle no pbc
assert abs(atoms.get_angle(1, 0, 2) - 104) < 1e-3

atoms.set_cell([2, 2, 2])

# Across different pbcs
atoms.set_pbc([True, False, True])
atoms.wrap()
assert abs(atoms.get_angle(1, 0, 2, mic=True) - 104) < 1e-3

# Across all True pbc
atoms.set_pbc(True)
atoms.wrap()
assert abs(atoms.get_angle(1, 0, 2, mic=True) - 104) < 1e-3

# Change Angle
old = atoms.get_angle(1, 0, 2, mic=False)
atoms.set_angle(1, 0, 2, -10, indices=[2], add=True)
new = atoms.get_angle(1, 0, 2, mic=False)
diff = old - new - 10
assert abs(diff) < 10e-3

#don't actually change angle using indices
old = atoms.get_angle(1, 0, 2, mic=False)
atoms.set_angle(1, 0, 2, -10, indices=[2, 1], add=True)
new = atoms.get_angle(1, 0, 2, mic=False)
diff = old - new
assert abs(diff) < 10e-3

# Simple tetrahedron
tetra_pos = np.array([[0, 0, 0], [1, 0, 0], [.5, np.sqrt(3) * .5, 0],
                      [.5, np.sqrt(1/3.) * .5, np.sqrt(2/3.)]])
atoms = Atoms(['H', 'H', 'H', 'H'],
              positions=tetra_pos - np.array([.2, 0, 0]))
angle = 70.5287793655
assert abs(atoms.get_dihedral(0, 1, 2, 3) - angle) < 1e-3

atoms.set_cell([3, 3, 3])
atoms.set_pbc(True)
atoms.wrap()
assert abs(atoms.get_dihedral(0, 1, 2, 3, mic=True) - angle) < 1e-3


