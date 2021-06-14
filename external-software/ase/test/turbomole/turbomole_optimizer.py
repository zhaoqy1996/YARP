from ase.calculators.turbomole import Turbomole
from ase.build import molecule
water = molecule('H2O')
params = {
    'title': 'water',
    'basis set name': 'sto-3g hondo',
    'total charge': 0,
    'multiplicity': 1,
    'use dft': True,
    'density functional': 'b-p',
    'use resolution of identity': True,
}

calc = Turbomole(**params)
optimizer = calc.get_optimizer(water)
optimizer.run(fmax=0.01, steps=5)

