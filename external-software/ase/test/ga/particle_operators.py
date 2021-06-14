from ase.cluster import Icosahedron
from ase.ga.particle_crossovers import CutSpliceCrossover
from random import shuffle

ico1 = Icosahedron('Cu', 3)
ico1.info['confid'] = 1
ico2 = Icosahedron('Ni', 3)
ico2.info['confid'] = 2

# TODO: Change this crossover to one for fixed particles
# op = CutSpliceCrossover({(28, 29): 2.0, (28, 28): 2.0, (29, 29): 2.0},
#                         keep_composition=False)
# a3, desc = op.get_new_individual([ico1, ico2])

# assert len(set(a3.get_chemical_symbols())) == 2
# assert len(a3) == 55

ico1.numbers[:20] = [28] * 20
shuffle(ico1.numbers)
ico2.numbers[:35] = [29] * 35
shuffle(ico2.numbers)
op = CutSpliceCrossover({(28, 29): 2.0, (28, 28): 2.0, (29, 29): 2.0})
a3, desc = op.get_new_individual([ico1, ico2])

assert a3.get_chemical_formula() == 'Cu35Ni20'

from ase.ga.particle_mutations import COM2surfPermutation
# from ase.ga.particle_mutations import RandomPermutation
# from ase.ga.particle_mutations import Poor2richPermutation
# from ase.ga.particle_mutations import Rich2poorPermutation

op = COM2surfPermutation(min_ratio=0.05)
a3, desc = op.get_new_individual([ico1])
a3.info['confid'] = 3

assert a3.get_chemical_formula() == 'Cu35Ni20'

aconf = op.get_atomic_configuration(a3)
core = aconf[1]
shell = aconf[-1]
for i, sym in zip(core, 6 * ['Ni'] + 6 * ['Cu']):
    a3[i].symbol = sym
for i, sym in zip(shell, 6 * ['Ni'] + 6 * ['Cu']):
    a3[i].symbol = sym

atomic_conf = op.get_atomic_configuration(a3, elements=['Cu'])[-2:]
cu3 = len([item for sublist in atomic_conf for item in sublist])
a4, desc = op.get_new_individual([a3])
atomic_conf = op.get_atomic_configuration(a4, elements=['Cu'])[-2:]
cu4 = len([item for sublist in atomic_conf for item in sublist])

assert abs(cu4 - cu3) == 1
