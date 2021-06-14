from __future__ import division
import numpy as np
from ase.geometry import cell_to_cellpar as c2p, cellpar_to_cell as p2c

# Make sure we get exactly zeros off-diagonal:
assert (p2c([1, 1, 1, 90, 90, 90]) == np.eye(3)).all()

eps = 2 * np.spacing(90., dtype=np.float64)


def nearly_equal(a, b):
    return np.all(np.abs(b - a) < eps)


def assert_equal(a, b):
    if not nearly_equal(a, b):
        msg = 'this:\n'
        msg += repr(a)
        msg += '\nand that:\n'
        msg += repr(b)
        msg += '\nwere supposed to be equal but are not.'
        raise AssertionError(msg)


# Constants
a = 5.43
d = a / 2.0
h = a / np.sqrt(2.0)


# Systems
# Primitive cell, non-orthorhombic, non-cubic
# Parameters
si_prim_p = np.array([h] * 3 + [60.] * 3)
# Tensor format
si_prim_m = np.array([[0., d, d],
                      [d, 0., d],
                      [d, d, 0.]])
# Tensor format in the default basis
si_prim_m2 = np.array([[2.0, 0., 0.],
                       [1.0, np.sqrt(3.0), 0.],
                       [1.0, np.sqrt(3.0) / 3.0, 2 * np.sqrt(2 / 3)]])
si_prim_m2 *= h / 2.0


# Orthorhombic cell, non-cubic
# Parameters
si_ortho_p = np.array([h] * 2 + [a] + [90.] * 3)
# Tensor format in the default basis
si_ortho_m = np.array([[h, 0.0, 0.0],
                       [0.0, h, 0.0],
                       [0.0, 0.0, a]])

# Cubic cell
# Parameters
si_cubic_p = np.array([a] * 3 + [90.] * 3)
# Tensor format in the default basis
si_cubic_m = np.array([[a, 0.0, 0.0],
                       [0.0, a, 0.0],
                       [0.0, 0.0, a]])

# Cell matrix -> cell parameters
assert_equal(c2p(si_prim_m), si_prim_p)
assert_equal(c2p(si_prim_m2), si_prim_p)
assert_equal(c2p(si_ortho_m), si_ortho_p)
assert_equal(c2p(si_cubic_m), si_cubic_p)
assert not nearly_equal(c2p(si_prim_m), si_ortho_p)

# Cell parameters -> cell matrix
assert_equal(p2c(si_prim_p), si_prim_m2)
assert_equal(p2c(si_ortho_p), si_ortho_m)
assert_equal(p2c(si_cubic_p), si_cubic_m)
assert not nearly_equal(p2c(si_prim_p), si_ortho_m)

# Idempotency (provided everything is provided in the default basis)
ref1 = si_prim_m2[:]
ref2 = si_ortho_m[:]
ref3 = si_cubic_m[:]
for i in range(20):
    ref1[:] = p2c(c2p(ref1))
    ref2[:] = p2c(c2p(ref2))
    ref3[:] = p2c(c2p(ref3))

assert_equal(ref1, si_prim_m2)
assert_equal(ref2, si_ortho_m)
assert_equal(ref3, si_cubic_m)
