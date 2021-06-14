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

    calc = CP2K()
    h2 = molecule('H2', calculator=calc)
    h2.center(vacuum=2.0)
    h2.get_potential_energy()
    calc.write('test_restart')  # write a restart
    calc2 = CP2K(restart='test_restart')  # load a restart
    assert not calc2.calculation_required(h2, ['energy'])
    print('passed test "restart"')


main()
