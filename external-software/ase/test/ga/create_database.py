from ase.ga.data import PrepareDB
from ase.ga.data import DataConnection
import os
import numpy as np

db_file = 'gadb.db'
if os.path.isfile(db_file):
    os.remove(db_file)


from ase.build import fcc111

atom_numbers = np.array([78, 78, 79, 79])
slab = fcc111('Ag', size=(4, 4, 2), vacuum=10.)

d = PrepareDB(db_file_name=db_file,
              simulation_cell=slab,
              stoichiometry=atom_numbers)

assert os.path.isfile(db_file)

dc = DataConnection(db_file)

slab_get = dc.get_slab()
an_get = dc.get_atom_numbers_to_optimize()

assert len(slab) == len(slab_get)
assert np.all(slab.numbers == slab_get.numbers)
assert np.all(slab.get_positions() == slab_get.get_positions())
assert np.all(an_get == atom_numbers)

os.remove(db_file)
