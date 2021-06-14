"""This test checks that the StrainFilter works using the default
built-in EMT calculator."""

import numpy as np
from ase.constraints import StrainFilter
from ase.optimize.mdmin import MDMin
from ase.calculators.emt import EMT
from ase.build import bulk

cu = bulk('Cu', 'fcc', a=3.6)

class EMTPlus(EMT):
    def get_stress(self, atoms):
        return np.zeros(6)

cu.set_calculator(EMTPlus())
f = StrainFilter(cu)
opt = MDMin(f, dt=0.01)
opt.run(0.1, steps=2)
