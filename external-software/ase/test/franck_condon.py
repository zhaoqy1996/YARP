from __future__ import print_function
import sys
import numpy as np

from ase.vibrations.franck_condon import FranckCondonOverlap, FranckCondonRecursive
from math import factorial


def equal(x, y, tolerance=0, fail=True, msg=''):
    """Compare x and y."""

    if not np.isfinite(x - y).any() or (np.abs(x - y) > tolerance).any():
        msg = (msg + '%s != %s (error: |%s| > %.9g)' %
               (x, y, x - y, tolerance))
        if fail:
            raise AssertionError(msg)
        else:
            sys.stderr.write('WARNING: %s\n' % msg)

# FCOverlap

fco = FranckCondonOverlap()
fcr = FranckCondonRecursive()

# check factorial
assert(fco.factorial(8) == factorial(8))
# the second test is useful according to the implementation
assert(fco.factorial(5) == factorial(5))
assert(fco.factorial.inv(5) == 1. / factorial(5))

# check T=0 and n=0 equality
S = np.array([1, 2.1, 34])
m = 5
assert(((fco.directT0(m, S) - fco.direct(0, m, S)) / fco.directT0(m, S) <
        1e-15).all())

# check symmetry
S = 2
n = 3
assert(fco.direct(n, m, S) == fco.direct(m, n, S))

# ---------------------------
# specials
S = np.array([0, 1.5])
delta = np.sqrt(2 * S)
for m in [2, 7]:
    equal(fco.direct0mm1(m, S)**2,
          fco.direct(1, m, S) * fco.direct(m, 0, S), 1.e-17)
    equal(fco.direct0mm1(m, S), fcr.ov0mm1(m, delta), 1.e-15)
    equal(fcr.ov0mm1(m, delta),
          fcr.ov0m(m, delta) * fcr.ov1m(m, delta), 1.e-15)
    equal(fcr.ov0mm1(m, -delta), fcr.direct0mm1(m, -delta), 1.e-15)
    equal(fcr.ov0mm1(m, delta), - fcr.direct0mm1(m, -delta), 1.e-15)

    equal(fco.direct0mm2(m, S)**2,
          fco.direct(2, m, S) * fco.direct(m, 0, S), 1.e-17)
    equal(fco.direct0mm2(m, S), fcr.ov0mm2(m, delta), 1.e-15)
    equal(fcr.ov0mm2(m, delta),
          fcr.ov0m(m, delta) * fcr.ov2m(m, delta), 1.e-15)
    equal(fco.direct0mm2(m, S), fcr.direct0mm2(m, delta), 1.e-15)

    equal(fcr.direct0mm3(m, delta),
          fcr.ov0m(m, delta) * fcr.ov3m(m, delta), 1.e-15)
    
    equal(fcr.ov1mm2(m, delta),
          fcr.ov1m(m, delta) * fcr.ov2m(m, delta), 1.e-15)
    equal(fcr.direct1mm2(m, delta), fcr.ov1mm2(m, delta), 1.e-15)

