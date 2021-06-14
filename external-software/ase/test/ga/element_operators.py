from ase import Atoms
from ase.ga.element_crossovers import OnePointElementCrossover

a1 = Atoms('SrSrSrBaClClClClBrBrBrBr')
a1.info['confid'] = 1
a2 = Atoms('CaCaMgBaFFFFFFFF')
a2.info['confid'] = 2

cations = ['Sr', 'Ba', 'Ca', 'Mg']
anions = ['Cl', 'F', 'Br']
op = OnePointElementCrossover([cations, anions],
                              [3, 2], [.25, .5])

a3, desc = op.get_new_individual([a1, a2])

syms = a3.get_chemical_symbols()
assert len(set([i for i in syms if i in cations])) < 4
assert len(set([i for i in syms if i in anions])) < 3

from ase.ga.element_mutations import RandomElementMutation

op = RandomElementMutation([cations, anions], [3, 2], [.25, .5])
a4, desc = op.get_new_individual([a1])
syms = a4.get_chemical_symbols()

assert len(set([i for i in syms if i in cations])) < 4
assert len(set([i for i in syms if i in anions])) < 3

op = RandomElementMutation(anions, 2, .5)
a4, desc = op.get_new_individual([a2])
syms = a4.get_chemical_symbols()

assert len(set([i for i in syms if i in anions])) == 2

from ase.ga.element_mutations import MoveDownMutation
from ase.ga.element_mutations import MoveUpMutation
from ase.ga.element_mutations import MoveRightMutation
from ase.ga.element_mutations import MoveLeftMutation

a1 = Atoms('SrSrClClClCl')
a1.info['confid'] = 1
op = MoveDownMutation(cations, 2, .5)
a2, desc = op.get_new_individual([a1])
a2.info['confid'] = 2

syms = a2.get_chemical_symbols()
assert 'Ba' in syms
assert len(set(syms)) == 3

op = MoveUpMutation(cations, 1, 1.)
a3, desc = op.get_new_individual([a2])
syms = a3.get_chemical_symbols()
assert 'Ba' not in syms
assert len(set(syms)) == 2

cations = ['Co', 'Ni', 'Cu']
a1 = Atoms('NiNiBrBr')
a1.info['confid'] = 1
op = MoveRightMutation(cations, 1, 1.)
a2, desc = op.get_new_individual([a1])
a2.info['confid'] = 2
syms = a2.get_chemical_symbols()

assert len(set(syms)) == 2
assert len([i for i in syms if i == 'Cu']) == 2

op = MoveLeftMutation(cations, 2, .5)
a3, desc = op.get_new_individual([a2])
syms = a3.get_chemical_symbols()

from ase.ga import set_raw_score, get_raw_score
assert len(set(syms)) == 3
set_raw_score(a3, 5.0)
assert get_raw_score(a3) == 5.0
