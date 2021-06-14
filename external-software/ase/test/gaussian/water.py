from ase.calculators.gaussian import Gaussian
from ase.atoms import Atoms
from ase.optimize.lbfgs import LBFGS


# First test to make sure Gaussian works
calc = Gaussian(method='pbepbe', basis='sto-3g', force='force',
                nproc=1, chk='water.chk', label='water')
calc.clean()

water = Atoms('OHH',
              positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
              calculator=calc)

opt = LBFGS(water)
opt.run(fmax=0.05)

forces = water.get_forces()
energy = water.get_potential_energy()
positions = water.get_positions()

# Then test the IO routines
from ase.io import read
water2 = read('water.log')
forces2 = water2.get_forces()
energy2 = water2.get_potential_energy()
positions2 = water2.get_positions()
#compare distances since positions are different in standard orientation
dist = water.get_all_distances()
dist2 = read('water.log', quantity='structures')[-1].get_all_distances()

assert abs(energy - energy2) < 1e-7
assert abs(forces - forces2).max() < 1e-9
assert abs(positions - positions2).max() < 1e-6
assert abs(dist - dist2).max() < 1e-6
