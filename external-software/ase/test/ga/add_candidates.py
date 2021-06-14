from ase.test import must_raise
from ase.build import fcc111
from ase.ga.data import PrepareDB
from ase.ga.data import DataConnection
from ase.ga.offspring_creator import OffspringCreator
from ase.ga import set_raw_score

import os

db_file = 'gadb.db'
if os.path.isfile(db_file):
    os.remove(db_file)

db = PrepareDB(db_file)

slab1 = fcc111('Ag', size=(2, 2, 2))
db.add_unrelaxed_candidate(slab1)

slab2 = fcc111('Cu', size=(2, 2, 2))
set_raw_score(slab2, 4)
db.add_relaxed_candidate(slab2)
assert slab2.info['confid'] == 3

db = DataConnection(db_file)
assert db.get_number_of_unrelaxed_candidates() == 1

slab3 = db.get_an_unrelaxed_candidate()
old_confid = slab3.info['confid']
slab3[0].symbol = 'Au'
db.add_unrelaxed_candidate(slab3, 'mutated: Parent {0}'.format(old_confid))
new_confid = slab3.info['confid']
# confid should update when using add_unrelaxed_candidate
assert old_confid != new_confid
slab3[1].symbol = 'Au'
db.add_unrelaxed_step(slab3, 'mutated: Parent {0}'.format(new_confid))
# confid should not change when using add_unrelaxed_step
assert slab3.info['confid'] == new_confid

with must_raise(AssertionError):
    db.add_relaxed_step(slab3)
set_raw_score(slab3, 3)
db.add_relaxed_step(slab3)

slab4 = OffspringCreator.initialize_individual(slab1,
                                               fcc111('Au', size=(2, 2, 2)))
set_raw_score(slab4, 67)
db.add_relaxed_candidate(slab4)
assert slab4.info['confid'] == 7

more_slabs = []
for m in ['Ni', 'Pd', 'Pt']:
    slab = fcc111(m, size=(2, 2, 2))
    slab = OffspringCreator.initialize_individual(slab1, slab)
    set_raw_score(slab, sum(slab.get_masses()))
    more_slabs.append(slab)
db.add_more_relaxed_candidates(more_slabs)
assert more_slabs[1].info['confid'] == 9

os.remove(db_file)
