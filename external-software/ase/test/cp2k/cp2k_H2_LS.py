# -*- coding: utf-8 -*-

"""Test suit for the CP2K ASE calulator.

http://www.cp2k.org
Author: Ole Schuett <ole.schuett@mat.ethz.ch>
"""

from __future__ import division, print_function
import os

from ase.test import NotAvailable
from ase.build import molecule
from ase.calculators.cp2k import CP2K


def main():
    if "ASE_CP2K_COMMAND" not in os.environ:
        raise NotAvailable('$ASE_CP2K_COMMAND not defined')

    inp = """&FORCE_EVAL
               &DFT
                 &QS
                   LS_SCF ON
                 &END QS
               &END DFT
             &END FORCE_EVAL"""
    calc = CP2K(label='test_H2_LS', inp=inp)
    h2 = molecule('H2', calculator=calc)
    h2.center(vacuum=2.0)
    energy = h2.get_potential_energy()
    energy_ref = -30.6989581747
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 5e-7
    print('passed test "H2_LS"')


main()
