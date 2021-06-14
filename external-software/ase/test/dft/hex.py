"""Test band structure from different variations of hexagonal cells."""
import numpy as np
from ase import Atoms
from ase.calculators.test import FreeElectrons
from ase.geometry import crystal_structure_from_cell
from ase.dft.kpoints import get_special_points

firsttime = True
for cell in [[[1, 0, 0], [0.5, 3**0.5 / 2, 0], [0, 0, 1]],
             [[1, 0, 0], [-0.5, 3**0.5 / 2, 0], [0, 0, 1]],
             [[0.5, -3**0.5 / 2, 0], [0.5, 3**0.5 / 2, 0], [0, 0, 1]]]:
    a = Atoms(cell=cell, pbc=True)
    a.cell *= 3
    a.calc = FreeElectrons(nvalence=1, kpts={'path': 'GMKG'})
    print(crystal_structure_from_cell(a.cell))
    r = a.get_reciprocal_cell()
    k = get_special_points(a.cell)['K']
    print(np.dot(k, r))
    a.get_potential_energy()
    bs = a.calc.band_structure()
    coords, labelcoords, labels = bs.get_labels()
    assert ''.join(labels) == 'GMKG'
    e_skn = bs.energies
    if firsttime:
        coords1 = coords
        labelcoords1 = labelcoords
        e_skn1 = e_skn
        firsttime = False
    else:
        for d in [coords - coords1,
                  labelcoords - labelcoords1,
                  e_skn - e_skn1]:
            assert abs(d).max() < 1e-13
    # bs.plot()
