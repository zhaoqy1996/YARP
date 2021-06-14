import numpy as np
from ase import Atoms
from ase.calculators.calculator import kptdensity2monkhorstpack as kd2mp
kd = 25 / (2 * np.pi)
a = 6.0
N = kd2mp(Atoms(cell=(a, a, a), pbc=True), kd)[0]
assert N * a / (2 * np.pi) >= kd, 'Too small k-point density'
