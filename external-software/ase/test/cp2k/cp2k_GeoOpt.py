# -*- coding: utf-8 -*-

"""Test suit for the CP2K ASE calulator.

http://www.cp2k.org
Author: Ole Schuett <ole.schuett@mat.ethz.ch>
"""

from __future__ import division, print_function
import os

from ase.test import NotAvailable
from ase.build import molecule
from ase.optimize import BFGS
from ase.calculators.cp2k import CP2K


def main():
    if "ASE_CP2K_COMMAND" not in os.environ:
        raise NotAvailable('$ASE_CP2K_COMMAND not defined')

    calc = CP2K(label='test_H2_GOPT', print_level='LOW')
    atoms = molecule('H2', calculator=calc)
    atoms.center(vacuum=2.0)

    # Run Geo-Opt
    gopt = BFGS(atoms, logfile=None)
    gopt.run(fmax=1e-6)

    # check distance
    dist = atoms.get_distance(0, 1)
    dist_ref = 0.7245595
    assert (dist - dist_ref) / dist_ref < 1e-7

    # check energy
    energy_ref = -30.7025616943
    energy = atoms.get_potential_energy()
    assert (energy - energy_ref) / energy_ref < 1e-10
    print('passed test "H2_GEO_OPT"')


main()
