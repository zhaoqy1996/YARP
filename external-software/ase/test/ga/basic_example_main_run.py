from ase.ga.data import PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.constraints import FixAtoms
import numpy as np
from ase.build import fcc111

db_file = 'gadb.db'

# create the surface
slab = fcc111('Au', size=(4, 4, 1), vacuum=10.0, orthogonal=True)
slab.set_constraint(FixAtoms(mask=len(slab) * [True]))

# define the volume in which the adsorbed cluster is optimized
# the volume is defined by a corner position (p0)
# and three spanning vectors (v1, v2, v3)
pos = slab.get_positions()
cell = slab.get_cell()
p0 = np.array([0., 0., max(pos[:, 2]) + 2.])
v1 = cell[0, :] * 0.8
v2 = cell[1, :] * 0.8
v3 = cell[2, :]
v3[2] = 3.

# Define the composition of the atoms to optimize
atom_numbers = 2 * [47] + 2 * [79]

# define the closest distance two atoms of a given species can be to each other
unique_atom_types = get_all_atom_types(slab, atom_numbers)
cd = closest_distances_generator(atom_numbers=unique_atom_types,
                                 ratio_of_covalent_radii=0.7)

# create the starting population
sg = StartGenerator(slab=slab,
                    atom_numbers=atom_numbers,
                    closest_allowed_distances=cd,
                    box_to_place_in=[p0, [v1, v2, v3]])

# generate the starting population
population_size = 5
starting_population = [sg.get_new_candidate() for i in range(population_size)]

# from ase.visualize import view   # uncomment these lines
# view(starting_population)        # to see the starting population

# create the database to store information in
d = PrepareDB(db_file_name=db_file,
              simulation_cell=slab,
              stoichiometry=atom_numbers)

for a in starting_population:
    d.add_unrelaxed_candidate(a)



from random import random
from ase.io import write
from ase.optimize import BFGS
from ase.calculators.emt import EMT

from ase.ga.data import DataConnection
from ase.ga.population import Population
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.ga.offspring_creator import OperationSelector
from ase.ga.standardmutations import MirrorMutation
from ase.ga.standardmutations import RattleMutation
from ase.ga.standardmutations import PermutationMutation
from ase.ga import set_raw_score

# Change the following three parameters to suit your needs
population_size = 5
mutation_probability = 0.3
n_to_test = 5

# Initialize the different components of the GA
da = DataConnection('gadb.db')
atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
n_to_optimize = len(atom_numbers_to_optimize)
slab = da.get_slab()
all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)
blmin = closest_distances_generator(all_atom_types,
                                    ratio_of_covalent_radii=0.7)

comp = InteratomicDistanceComparator(n_top=n_to_optimize,
                                     pair_cor_cum_diff=0.015,
                                     pair_cor_max=0.7,
                                     dE=0.02,
                                     mic=False)

pairing = CutAndSplicePairing(slab, n_to_optimize, blmin)
mutations = OperationSelector([1., 1., 1.],
                              [MirrorMutation(blmin, n_to_optimize),
                               RattleMutation(blmin, n_to_optimize),
                               PermutationMutation(n_to_optimize)])

# Relax all unrelaxed structures (e.g. the starting population)
while da.get_number_of_unrelaxed_candidates() > 0:
    a = da.get_an_unrelaxed_candidate()
    a.set_calculator(EMT())
    print('Relaxing starting candidate {0}'.format(a.info['confid']))
    dyn = BFGS(a, trajectory=None, logfile=None)
    dyn.run(fmax=0.05, steps=100)
    set_raw_score(a, -a.get_potential_energy())
    da.add_relaxed_step(a)

# create the population
population = Population(data_connection=da,
                        population_size=population_size,
                        comparator=comp)

# test n_to_test new candidates
for i in range(n_to_test):
    print('Now starting configuration number {0}'.format(i))
    a1, a2 = population.get_two_candidates()
    a3, desc = pairing.get_new_individual([a1, a2])
    if a3 is None:
        continue
    da.add_unrelaxed_candidate(a3, description=desc)

    # Check if we want to do a mutation
    if random() < mutation_probability:
        a3_mut, desc = mutations.get_new_individual([a3])
        if a3_mut is not None:
            da.add_unrelaxed_step(a3_mut, desc)
            a3 = a3_mut
        
    # Relax the new candidate
    a3.set_calculator(EMT())
    dyn = BFGS(a3, trajectory=None, logfile=None)
    dyn.run(fmax=0.05, steps=100)
    set_raw_score(a3, -a3.get_potential_energy())
    da.add_relaxed_step(a3)
    population.update()

write('all_candidates.traj', da.get_all_relaxed_candidates())
