from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator, atoms_too_close
from ase.ga.cutandsplicepairing import CutAndSplicePairing
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

sg = StartGenerator(slab=slab,
                    atom_numbers=atom_numbers,
                    closest_allowed_distances=cd,
                    box_to_place_in=[p0, [v1, v2, v3]])

c1 = sg.get_new_candidate()
c1.info['confid'] = 1
c2 = sg.get_new_candidate()
c2.info['confid'] = 2

n_top = len(atom_numbers)

pairing = CutAndSplicePairing(slab, n_top, cd)

c3, desc = pairing.get_new_individual([c1, c2])

# verify that the stoichiometry is preserved
assert np.all(c3.numbers == c1.numbers)
top1 = c1[-n_top:]
top2 = c2[-n_top:]
top3 = c3[-n_top:]


# verify that the positions in the new candidate come from c1 or c2
n1 = -1 * np.ones((n_top, ))
n2 = -1 * np.ones((n_top, ))
for i in range(n_top):
    for j in range(n_top):
        if np.all(top1.positions[j, :] == top3.positions[i, :]):
            n1[i] = j
            break
        elif np.all(top2.positions[j, :] == top3.positions[i, :]):
            n2[i] = j
            break
    assert (n1[i] > -1 and n2[i] == -1) or (n1[i] == -1 and n2[i] > -1)

# verify that c3 includes atoms from both c1 and c2
assert len(n1[n1 > -1]) > 0 and len(n2[n2 > -1]) > 0

# verify no atoms too close
assert not atoms_too_close(top3, cd)
