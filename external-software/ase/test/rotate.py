import numpy as np
from math import sqrt
from ase import Atoms
from ase.utils import rotate, irotate


def test(xyz):
    a = rotate(xyz)
    ixyz = '%sx,%sy,%sz' % irotate(a)
    a2 = rotate(ixyz)
    print(xyz)
    print(ixyz)
    assert abs(a - a2).max() < 1e-10

test('10z')
test('155x,43y,190z')
test('55x,90y,190z')
test('180x,-90y,45z')
test('-180y')
test('40z,50x')

norm = np.linalg.norm

for eps in [1.e-6, 1.e-8]:
    struct = Atoms('H2',
                   [[0, 0, 0],
                    [0, sqrt(1 - eps**2), eps]])
    struct.rotate(struct[1].position, 'y')
    assert abs(norm(struct[1].position) - 1) < 1.e-12
