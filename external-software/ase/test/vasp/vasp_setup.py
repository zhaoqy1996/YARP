"""
Run some tests to ensure that VASP calculator constructs correct POTCAR files

"""

from os import remove
from os.path import isfile
from ase.atoms import Atoms
from ase.calculators.vasp import Vasp


def check_potcar(setups, filename='POTCAR'):
    """Return true if labels in setups are found in POTCAR"""

    pp = []
    with open(filename, 'r') as f:
        for line in f:
            if 'TITEL' in line.split():
                pp.append(line.split()[3])
    for setup in setups:
        assert setup in pp

# Write some POTCARs and check they are ok
potcar = 'POTCAR'
try:
    atoms = Atoms('CaGdCs',
                  positions=[[0, 0, 1], [0, 0, 2], [0, 0, 3]], cell=[5, 5, 5])

    calc = Vasp(xc='pbe')
    calc.initialize(atoms)
    calc.write_potcar()
    check_potcar(('Ca_pv', 'Gd', 'Cs_sv'), filename=potcar)

    calc = Vasp(xc='pbe', setups='recommended')
    calc.initialize(atoms)
    calc.write_potcar()
    check_potcar(('Ca_sv', 'Gd_3', 'Cs_sv'), filename=potcar)

    atoms = Atoms('CaInI',
                  positions=[[0, 0, 1], [0, 0, 2], [0, 0, 3]], cell=[5, 5, 5])
    calc = Vasp(xc='pbe', setups={'base': 'gw'})
    calc.initialize(atoms)
    calc.write_potcar()
    check_potcar(('Ca_sv_GW', 'In_d_GW', 'I_GW'), filename=potcar)

    calc = Vasp(xc='pbe', setups={'base': 'gw', 'I': ''})
    calc.initialize(atoms)
    calc.write_potcar()
    check_potcar(('Ca_sv_GW', 'In_d_GW', 'I'), filename=potcar)

    calc = Vasp(xc='pbe', setups={'base': 'gw', 'Ca': '_sv', 2: 'I'})
    calc.initialize(atoms)
    calc.write_potcar()
    check_potcar(('Ca_sv', 'In_d_GW', 'I'), filename=potcar)
finally:
    if isfile(potcar):
        remove(potcar)
