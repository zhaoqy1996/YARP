import os

from ase.calculators.calculator import get_calculator
from ase.io import read, write
from ase.build import molecule
from ase.test import test_calculator_names


def h2(name, par):
    h2 = molecule('H2', pbc=par.pop('pbc', False))
    h2.center(vacuum=2.0)
    h2.calc = get_calculator(name)(**par)
    e = h2.get_potential_energy()
    assert not h2.calc.calculation_required(h2, ['energy'])
    f = h2.get_forces()
    assert not h2.calc.calculation_required(h2, ['energy', 'forces'])
    write('h2.traj', h2)
    h2 = read('h2.traj')
    assert abs(e - h2.get_potential_energy()) < 1e-12
    assert abs(f - h2.get_forces()).max() < 1e-12


parameters = {
    'abinit': dict(ecut=200, toldfe=0.0001),
    'aims': dict(sc_accuracy_rho=5.e-3, sc_accuracy_forces=1e-4, xc='LDA'),
    'gpaw': dict(mode={'name': 'lcao', 'interpolation': 'fft'},
                 basis='sz(dzp)'),
    'elk': dict(tasks=0, rgkmax=5.0, epsengy=1.0, epspot=1.0, tforce=True,
                pbc=True),
    'jacapo': dict(pbc=True),
    'vasp': dict(xc='LDA'),
    'espresso': dict(pbc=True, tprnfor=True,
                     pseudopotentials={'H': 'H.pbe-rrkjus_psl.0.1.UPF'})}

for name in test_calculator_names + ['emt']:
    if name in ['cp2k', 'gromacs', 'lammpslib', 'lammpsrun', 'mopac', 'turbomole', 'amber', 'asap']:
        continue
    par = parameters.get(name, {})
    os.mkdir(name + '-test')
    os.chdir(name + '-test')
    h2(name, par)
    os.chdir('..')
