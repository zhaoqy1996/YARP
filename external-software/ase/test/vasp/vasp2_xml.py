"""
Run some VASP tests to ensure that the VASP calculator works. This
is conditional on the existence of the VASP_COMMAND or VASP_SCRIPT
environment variables

"""

from ase.test import NotAvailable
from ase.test.vasp import installed2 as installed
from ase import Atoms
from ase.calculators.vasp import Vasp2 as Vasp
from ase.io import read
import numpy as np
import sys

assert installed()


def main():
    if sys.version_info < (2, 7):
        raise NotAvailable('read_xml requires Python version 2.7 or greater')

    assert installed()

    # simple test calculation of CO molecule
    d = 1.14
    co = Atoms('CO', positions=[(0, 0, 0), (0, 0, d)],
               pbc=True)
    co.center(vacuum=5.)

    calc = Vasp(xc='PBE',
                prec='Low',
                algo='Fast',
                ismear=0,
                sigma=1.,
                istart=0,
                lwave=False,
                lcharg=False,
                ldipol=True)

    co.set_calculator(calc)
    energy = co.get_potential_energy()
    forces = co.get_forces()
    dipole_moment = co.get_dipole_moment()

    # check that parsing of vasprun.xml file works
    conf = read('vasprun.xml')
    assert conf.calc.parameters['kpoints_generation']
    assert conf.calc.parameters['sigma'] == 1.0
    assert conf.calc.parameters['ialgo'] == 68
    assert energy - conf.get_potential_energy() == 0.0
    assert np.allclose(conf.get_forces(), forces)
    assert np.allclose(conf.get_dipole_moment(), dipole_moment, atol=1e-6)

    # Cleanup
    calc.clean()


if 1:
    main()
