""" test run for DFTB+ calculator
    the tolerance is extremely loose, beause different sk files
    give different results

"""
from ase.calculators.dftb import Dftb
from ase.optimize import QuasiNewton
from ase.build import molecule

test = molecule('H2O')
test.calc = Dftb(label='h2o')

dyn = QuasiNewton(test, trajectory='test.traj')
dyn.run(fmax=0.01)
final_energy = test.get_potential_energy()
assert abs(final_energy + 111.141945) < 1.0
