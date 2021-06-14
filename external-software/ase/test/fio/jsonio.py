"""Test serialization of ndarrays and other stuff."""

import numpy as np

from ase.io.jsonio import encode, decode


assert decode(encode(np.int64(42))) == 42

c = np.array([0.1j])
assert (decode(encode(c)) == c).all()
