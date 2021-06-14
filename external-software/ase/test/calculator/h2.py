from ase.test import NotAvailable
from ase.build import molecule
from ase.calculators.calculator import get_calculator


required = {'abinit': dict(ecut=200, toldfe=0.0001),
            'aims': dict(sc_accuracy_rho=5.e-3),
            'gpaw': dict(mode='lcao', basis='sz(dzp)', realspace=False),
            'cp2k': dict(auto_write=True, uks=True)}


def h2dft(name):
    Calculator = get_calculator(name)
    par = required.get(name, {})
    calc = Calculator(label=name, xc='LDA', **par)
    h2 = molecule('H2', calculator=calc)
    h2.center(vacuum=2.0)
    e2 = h2.get_potential_energy()
    calc.set(xc='PBE')
    e2pbe = h2.get_potential_energy()
    h1 = h2.copy()
    del h1[1]
    h1.set_initial_magnetic_moments([1])
    h1.calc = calc
    e1pbe = h1.get_potential_energy()
    calc.set(xc='LDA')
    e1 = h1.get_potential_energy()
    try:
        m1 = h1.get_magnetic_moment()
    except NotImplementedError:
        pass
    else:
        print(m1)
    print(2 * e1 - e2)
    print(2 * e1pbe - e2pbe)
    print(e1, e2, e1pbe, e2pbe)
    calc = Calculator(name)
    print(calc.parameters, calc.results, calc.atoms)
    assert not calc.calculation_required(h1, ['energy'])
    h1 = calc.get_atoms()
    print(h1.get_potential_energy())
    label = 'dir/' + name + '-h1'
    calc = Calculator(label=label, atoms=h1, xc='LDA', **par)
    print(h1.get_potential_energy())
    print(Calculator.read_atoms(label).get_potential_energy())

names = ['abinit', 'aims', 'gaussian', 'nwchem', 'cp2k']
for name in names:
    try:
        h2dft(name)
    except NotAvailable:
        pass
