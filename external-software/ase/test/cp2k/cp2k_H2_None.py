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

inp = """
&FORCE_EVAL
   METHOD Quickstep
   &DFT
      BASIS_SET_FILE_NAME BASIS_MOLOPT
      &MGRID
         CUTOFF 400
      &END MGRID
      &XC
         &XC_FUNCTIONAL LDA
         &END XC_FUNCTIONAL
      &END XC
      &POISSON
         PERIODIC NONE
         PSOLVER  MT
      &END POISSON
   &END DFT
     &SUBSYS
      &KIND H
         BASIS_SET DZVP-MOLOPT-SR-GTH
         POTENTIAL GTH-LDA
      &END KIND
   &END SUBSYS
&END FORCE_EVAL
"""


def main():
    if "ASE_CP2K_COMMAND" not in os.environ:
        raise NotAvailable('$ASE_CP2K_COMMAND not defined')

    # Basically, the entire CP2K input is passed in explicitly.
    # Disable ASE's input generation by setting everything to None.
    # ASE should only add the CELL and the COORD section.
    calc = CP2K(basis_set=None,
                basis_set_file=None,
                max_scf=None,
                cutoff=None,
                force_eval_method=None,
                potential_file=None,
                poisson_solver=None,
                pseudo_potential=None,
                stress_tensor=False,
                xc=None,
                label='test_H2_inp', inp=inp)
    h2 = molecule('H2', calculator=calc)
    h2.center(vacuum=2.0)
    energy = h2.get_potential_energy()
    energy_ref = -30.6989595886
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 1e-10
    print('passed test "H2_None"')


main()
