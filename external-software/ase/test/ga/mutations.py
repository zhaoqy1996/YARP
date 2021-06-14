from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.ga.standardmutations import RattleMutation, PermutationMutation
import numpy as np
from ase.build import fcc111
from ase.constraints import FixAtoms

# first create two random starting candidates
slab = fcc111('Au', size=(4, 4, 2), vacuum=10.0, orthogonal=True)
slab.set_constraint(FixAtoms(mask=slab.positions[:, 2] <= 10.))

pos = slab.get_positions()
cell = slab.get_cell()
p0 = np.array([0., 0., max(pos[:, 2]) + 2.])
v1 = cell[0, :] * 0.8
v2 = cell[1, :] * 0.8
v3 = cell[2, :]
v3[2] = 3.

cd = closest_distances_generator(atom_numbers=[47, 79],
                                 ratio_of_covalent_radii=0.7)

atom_numbers = 2 * [47] + 2 * [79]
n_top = len(atom_numbers)
sg = StartGenerator(slab=slab,
                    atom_numbers=atom_numbers,
                    closest_allowed_distances=cd,
                    box_to_place_in=[p0, [v1, v2, v3]])

c1 = sg.get_new_candidate()
c1.info['confid'] = 1

# first verify that the rattle mutation works
rmut = RattleMutation(cd, n_top, rattle_strength=0.8, rattle_prop=0.4)

c2, desc = rmut.get_new_individual([c1])

assert np.all(c1.numbers == c2.numbers)

top1 = c1[-n_top:]
top2 = c2[-n_top:]
slab2 = c2[0:(len(c1) - n_top)]

assert len(slab) == len(slab2)
assert np.all(slab.get_positions() == slab2.get_positions())

dp = np.sum((top2.get_positions() - top1.get_positions())**2, axis=1)**0.5

# check that all displacements are smaller than the rattle strength we
# cannot check if 40 % of the structures have been rattled since it is
# probabilistic and because the probability will be lower if two atoms
# get too close
for p in dp:
    assert p < 0.8 * 3**0.5

# now we check the permutation mutation

mmut = PermutationMutation(n_top, probability=0.5)

c3, desc = mmut.get_new_individual([c1])
assert np.all(c1.numbers == c3.numbers)

top1 = c1[-n_top:]
top2 = c3[-n_top:]
slab2 = c3[0:(len(c1) - n_top)]

assert len(slab) == len(slab2)
assert np.all(slab.get_positions() == slab2.get_positions())
dp = np.sum((top2.get_positions() - top1.get_positions())**2, axis=1)**0.5

# verify that two positions have been changed
assert len(dp[dp > 0]) == 2
