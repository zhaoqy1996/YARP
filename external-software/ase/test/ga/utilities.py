from __future__ import division
import numpy as np

from ase.cluster import Icosahedron
from ase.calculators.emt import EMT
from ase.optimize.fire import FIRE
from ase.lattice.compounds import L1_2

from ase.ga.utilities import get_rdf

eps = 1e-5

atoms = Icosahedron('Cu', 3)
atoms.numbers[[0, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30]] = 79
atoms.set_calculator(EMT())
opt = FIRE(atoms, logfile=None)
opt.run(fmax=0.05)

rmax = 8.
nbins = 5
rdf, dists = get_rdf(atoms, rmax, nbins)
calc_dists = np.arange(rmax / (2 * nbins), rmax, rmax / nbins)
assert all(abs(dists - calc_dists) < eps)
calc_rdf = [0., 0.84408157, 0.398689, 0.23748934, 0.15398546]
assert all(abs(rdf - calc_rdf) < eps)

dm = atoms.get_all_distances()
s = np.zeros(5)
for c in [(29, 29), (29, 79), (79, 29), (79, 79)]:
    inv_norm = len(np.where(atoms.numbers == c[0])[0]) / len(atoms)
    s += get_rdf(atoms, rmax, nbins, elements=c,
                 distance_matrix=dm, no_dists=True) * inv_norm
assert all(abs(s - calc_rdf) < eps)

AuAu = get_rdf(atoms, rmax, nbins, elements=(79, 79),
               distance_matrix=dm, no_dists=True)
assert all(abs(AuAu[-2:] - [0.12126445, 0.]) < eps)

bulk = L1_2(['Au', 'Cu'], size=(2, 2, 2), latticeconstant=np.sqrt(2))
dm = bulk.get_all_distances(mic=True)
rdf = get_rdf(bulk, 5., 3, distance_matrix=dm)[0]
calc_rdf = [0.54694216, 0.08334357, 0.]
assert all(abs(rdf - calc_rdf) < eps)