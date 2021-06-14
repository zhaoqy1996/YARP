"Test that set_angle() and get_angle() in Atoms are consistent"

from ase import Atoms

atoms = Atoms('HHCCHH', [[-1, 1, 0], [-1, -1, 0], [0, 0, 0],
                         [1, 0, 0], [2, 1, 0], [2, -1, 0]])

list = [2, 3, 4]
theta = 20
old_angle = atoms.get_angle(*list)
atoms.set_angle(*list, angle=old_angle + theta)
new_angle = atoms.get_angle(*list)

assert abs(new_angle - (old_angle + theta)) < 1.0e-9
