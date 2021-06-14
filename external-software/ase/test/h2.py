from ase import Atoms
from ase.calculators.emt import EMT

h2 = Atoms('H2', positions=[(0, 0, 0), (0, 0, 1.1)],
           calculator=EMT())
f1 = h2.calc.calculate_numerical_forces(h2, 0.0001)
f2 = h2.get_forces()
assert abs(f1 - f2).max() < 1e-6
