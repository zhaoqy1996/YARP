"""Test that the filter and trajectories are playing well together."""

from ase.build import molecule
from ase.constraints import Filter
from ase.optimize import QuasiNewton
from ase.calculators.emt import EMT

atoms = molecule('CO2')
atoms.set_calculator(EMT())
filter = Filter(atoms, indices=[1, 2])

opt = QuasiNewton(filter, trajectory='filter-test.traj', logfile='filter-test.log')
opt.run()
