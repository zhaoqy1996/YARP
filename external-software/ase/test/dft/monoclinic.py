"""Test band structure from different variations of hexagonal cells."""
import numpy as np
from ase import Atoms
from ase.calculators.test import FreeElectrons
from ase.geometry import (crystal_structure_from_cell, cell_to_cellpar,
                          cellpar_to_cell)
from ase.dft.kpoints import get_special_points

mc1 = [[1, 0, 0], [0, 1, 0], [0, 0.2, 1]]
par = cell_to_cellpar(mc1)
mc2 = cellpar_to_cell(par)
mc3 = [[1, 0, 0], [0, 1, 0], [-0.2, 0, 1]]
mc4 = [[1, 0, 0], [-0.2, 1, 0], [0, 0, 1]]
path = 'GYHCEM1AXH1'

firsttime = True
for cell in [mc1, mc2, mc3, mc4]:
    a = Atoms(cell=cell, pbc=True)
    a.cell *= 3
    a.calc = FreeElectrons(nvalence=1, kpts={'path': path})
    cs = crystal_structure_from_cell(a.cell)
    assert cs == 'monoclinic'
    r = a.get_reciprocal_cell()
    k = get_special_points(a.cell)['H']
    print(np.dot(k, r))
    a.get_potential_energy()
    bs = a.calc.band_structure()
    coords, labelcoords, labels = bs.get_labels()
    assert ''.join(labels) == path
    e_skn = bs.energies
    # bs.plot()
    if firsttime:
        coords1 = coords
        labelcoords1 = labelcoords
        e_skn1 = e_skn
        firsttime = False
    else:
        for d in [coords - coords1,
                  labelcoords - labelcoords1,
                  e_skn - e_skn1]:
            print(abs(d).max())
            assert abs(d).max() < 1e-13, d
