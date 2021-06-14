"""

Check the many ways of specifying KPOINTS

"""

import os
import filecmp

from ase.calculators.vasp import Vasp2 as Vasp
from ase.build import bulk
from ase.test.vasp import installed2 as installed

assert installed()


Al = bulk('Al', 'fcc', a=4.5, cubic=True)


def check_kpoints_line(n, contents):
    """Assert the contents of a line"""
    with open('KPOINTS', 'r') as f:
        lines = f.readlines()
        assert lines[n] == contents

# Default to (1 1 1)

calc = Vasp(gamma=True)
calc.write_kpoints()
check_kpoints_line(2, 'Gamma\n')
check_kpoints_line(3, '1 1 1 \n')
calc.clean()

# 3-tuple prints mesh
calc = Vasp(gamma=False, kpts=(4, 4, 4))
calc.write_kpoints()
check_kpoints_line(2, 'Monkhorst-Pack\n')
check_kpoints_line(3, '4 4 4 \n')
calc.clean()

# Auto mode
calc = Vasp(kpts=20)
calc.write_kpoints()
check_kpoints_line(1, '0\n')
check_kpoints_line(2, 'Auto\n')
check_kpoints_line(3, '20 \n')
calc.clean()

# 1-element list ok, Gamma ok
calc = Vasp(kpts=[20], gamma=True)
calc.write_kpoints()
check_kpoints_line(1, '0\n')
check_kpoints_line(2, 'Auto\n')
check_kpoints_line(3, '20 \n')
calc.clean()

# KSPACING suppresses KPOINTS file
calc = Vasp(kspacing=0.23)
calc.initialize(Al)
calc.write_kpoints()
calc.write_incar(Al)
assert not os.path.isfile('KPOINTS')
with open('INCAR', 'r') as f:
    assert ' KSPACING = 0.230000\n' in f.readlines()
calc.clean()

# Negative KSPACING raises an error
calc = Vasp(kspacing=-0.5)

try:
    calc.write_kpoints()
except ValueError:
    pass
else:
    raise AssertionError("Negative KSPACING did not raise ValueError")
calc.clean()

# Explicit weighted points with nested lists, Cartesian if not specified
calc = Vasp(
    kpts=[[0.1, 0.2, 0.3, 2], [0.0, 0.0, 0.0, 1], [0.0, 0.5, 0.5, 2]])
calc.write_kpoints()

with open('KPOINTS.ref', 'w') as f:
    f.write("""KPOINTS created by Atomic Simulation Environment
3 
Cartesian
0.100000 0.200000 0.300000 2.000000 
0.000000 0.000000 0.000000 1.000000 
0.000000 0.500000 0.500000 2.000000 
""")

assert filecmp.cmp('KPOINTS', 'KPOINTS.ref')
os.remove('KPOINTS.ref')

# Explicit points as list of tuples, automatic weighting = 1.
calc = Vasp(
    kpts=[(0.1, 0.2, 0.3), (0.0, 0.0, 0.0), (0.0, 0.5, 0.5)], reciprocal=True)
calc.write_kpoints()

with open('KPOINTS.ref', 'w') as f:
    f.write("""KPOINTS created by Atomic Simulation Environment
3 
Reciprocal
0.100000 0.200000 0.300000 1.0 
0.000000 0.000000 0.000000 1.0 
0.000000 0.500000 0.500000 1.0 
""")

assert filecmp.cmp('KPOINTS', 'KPOINTS.ref')
os.remove('KPOINTS.ref')
