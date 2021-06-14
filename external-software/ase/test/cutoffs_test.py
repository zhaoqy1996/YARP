from ase.utils import natural_cutoffs
from ase import Atoms
import numpy as np

atoms = Atoms("HCOPtAu")

assert np.allclose(natural_cutoffs(atoms), [0.31, 0.76, 0.66, 1.36, 1.36])
assert np.allclose(natural_cutoffs(atoms, mult=1.2), [0.372, 0.912, 0.792,
                                                      1.632, 1.632])
assert np.allclose(natural_cutoffs(atoms, mult=1.2, Au=1), [0.372, 0.912,
                                                            0.792, 1.632, 1])
