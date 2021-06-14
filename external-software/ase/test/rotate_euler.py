from math import sqrt
from ase import Atoms

d = 1.14
a = Atoms('CO', [(0, 0, 0), (d, 0, 0)])
a.euler_rotate(phi=90, theta=45, psi=180)
for p in a[0].position:
    assert p == 0.0
assert abs(a[1].position[0]) < 1e-15
d2 = d / sqrt(2)
assert abs(a[1].position[1] - d2) < 1e-15
assert abs(a[1].position[2] - d2) < 1e-15
