from __future__ import print_function
from ase.cluster.cubic import FaceCenteredCubic
from ase.calculators.turbomole import Turbomole

surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
layers = [1, 2, 1]
atoms = FaceCenteredCubic('Au', surfaces, layers, latticeconstant=4.08)

params = {
    'title': 'Au13-',
    'task': 'energy',
    'basis set name': 'def2-SV(P)',
    'total charge': -1,
    'multiplicity': 1,
    'use dft': True,
    'density functional': 'pbe',
    'use resolution of identity': True,
    'ri memory': 1000,
    'use fermi smearing': True,
    'fermi initial temperature': 500,
    'fermi final temperature': 100,
    'fermi annealing factor': 0.9,
    'fermi homo-lumo gap criterion': 0.09,
    'fermi stopping criterion': 0.002,
    'scf energy convergence': 1.e-4,
    'scf iterations': 250
}

calc = Turbomole(**params)
atoms.set_calculator(calc)
calc.calculate(atoms)

# use the get_property() method
print(calc.get_property('energy'))
print(calc.get_property('dipole'))

# test restart

params = {
    'task': 'gradient',
    'scf energy convergence': 1.e-6
}

calc = Turbomole(restart=True, **params)
assert calc.converged
calc.calculate()

print(calc.get_property('energy'))
print(calc.get_property('forces'))
print(calc.get_property('dipole'))
