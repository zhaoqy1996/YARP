import warnings

import numpy as np

from ase.io import read, write
from ase.calculators.emt import EMT
from ase.build import bulk

at = bulk("Cu")
at.rattle()
at.set_calculator(EMT())
f = at.get_forces()

write("tmp.xyz", at)
at2 = read("tmp.xyz")
f2 = at.get_forces()

assert np.abs(f - f2).max() < 1e-6

with warnings.catch_warnings(record=True) as w:
    # Cause all warnings to always be triggered.
    warnings.simplefilter("always")
    write("tmp2.xyz", at2)
    assert len(w) == 1
    assert ('overwriting array' in str(w[0].message))

at3 = read("tmp2.xyz")
f3 = at3.get_forces()
assert np.abs(f - f3).max() < 1e-6
