# -*- coding: utf-8 -*-

"""Test suit for the CP2K ASE calulator.

http://www.cp2k.org
Author: Ole Schuett <ole.schuett@mat.ethz.ch>
"""

from __future__ import division, print_function
import numpy as np
import os

from ase.test import NotAvailable
from ase.build import bulk
from ase.constraints import UnitCellFilter
from ase.optimize import MDMin
from ase.calculators.cp2k import CP2K


def main():
    """Adopted from ase/test/stress.py"""

    if "ASE_CP2K_COMMAND" not in os.environ:
        raise NotAvailable('$ASE_CP2K_COMMAND not defined')

    # setup a Fist Lennard-Jones Potential
    inp = """&FORCE_EVAL
                  &MM
                    &FORCEFIELD
                      &SPLINE
                        EMAX_ACCURACY 500.0
                        EMAX_SPLINE    1000.0
                        EPS_SPLINE 1.0E-9
                      &END
                      &NONBONDED
                        &LENNARD-JONES
                          atoms Ar Ar
                          EPSILON [eV] 1.0
                          SIGMA [angstrom] 1.0
                          RCUT [angstrom] 10.0
                        &END LENNARD-JONES
                      &END NONBONDED
                      &CHARGE
                        ATOM Ar
                        CHARGE 0.0
                      &END CHARGE
                    &END FORCEFIELD
                    &POISSON
                      &EWALD
                        EWALD_TYPE none
                      &END EWALD
                    &END POISSON
                  &END MM
                &END FORCE_EVAL"""

    calc = CP2K(label="test_stress", inp=inp, force_eval_method="Fist")

    # Theoretical infinite-cutoff LJ FCC unit cell parameters
    vol0 = 4 * 0.91615977036  # theoretical minimum
    a0 = vol0 ** (1 / 3)

    a = bulk('Ar', 'fcc', a=a0)
    cell0 = a.get_cell()

    a.calc = calc
    a.set_cell(np.dot(a.cell,
                      [[1.02, 0, 0.03],
                       [0, 0.99, -0.02],
                       [0.1, -0.01, 1.03]]),
               scale_atoms=True)

    a *= (1, 2, 3)
    cell0 *= np.array([1, 2, 3])[:, np.newaxis]

    a.rattle()

    # Verify analytical stress tensor against numerical value
    s_analytical = a.get_stress()
    s_numerical = a.calc.calculate_numerical_stress(a, 1e-5)
    s_p_err = 100 * (s_numerical - s_analytical) / s_numerical

    print("Analytical stress:\n", s_analytical)
    print("Numerical stress:\n", s_numerical)
    print("Percent error in stress:\n", s_p_err)
    assert np.all(abs(s_p_err) < 1e-5)

    # Minimize unit cell
    opt = MDMin(UnitCellFilter(a), dt=0.01)
    opt.run(fmax=1e-3)

    # Verify minimized unit cell using Niggli tensors
    g_minimized = np.dot(a.cell, a.cell.T)
    g_theory = np.dot(cell0, cell0.T)
    g_p_err = 100 * (g_minimized - g_theory) / g_theory

    print("Minimized Niggli tensor:\n", g_minimized)
    print("Theoretical Niggli tensor:\n", g_theory)
    print("Percent error in Niggli tensor:\n", g_p_err)
    assert np.all(abs(g_p_err) < 1)

    print('passed test "stress"')


main()
# EOF
