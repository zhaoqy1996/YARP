import numpy as np
from ase.dft.bandgap import bandgap


class Calculator:
    def __init__(self, e_skn):
        self.e_skn = np.array(e_skn, dtype=float)
        self.ns, self.nk, self.nb = self.e_skn.shape

    def get_ibz_k_points(self):
        k = np.zeros((self.nk, 3))
        k[:, 0] += np.arange(self.nk)
        return k

    def get_fermi_level(self):
        return 0.0

    def get_eigenvalues(self, kpt, spin):
        return self.e_skn[spin, kpt]

    def get_number_of_spins(self):
        return self.ns


def test(e_skn):
    c = Calculator(e_skn)
    if c.ns == 1:
        result = [bandgap(c), bandgap(c, direct=True)]
        return [(gap, k1, k2) for gap, (s1, k1, n1), (s2, k2, n2) in result]
    result = [bandgap(c), bandgap(c, direct=True),
              bandgap(c, spin=0), bandgap(c, direct=True, spin=0),
              bandgap(c, spin=1), bandgap(c, direct=True, spin=1)]
    for gap, (s1, k1, n1), (s2, k2, n2) in result:
        if k1 is not None:
            assert gap == e_skn[s2][k2][n2] - e_skn[s1][k1][n1]
    return [(gap, (s1, k1), (s2, k2))
            for gap, (s1, k1, n1), (s2, k2, n2) in result]


r = test([[[-1, 1]]])
assert r == [(2, 0, 0), (2, 0, 0)]
r = test([[[-1, 2], [-3, 1]]])
assert r == [(2, 0, 1), (3, 0, 0)]
r = test([[[-1, 2, 3], [-1, -1, 1]]])
assert r == [(0, None, None), (0, None, None)]
r = test([[[-1, 2, 3], [-1, -1, 1]], [[-1, 2, 2], [-3, 1, 1]]])
assert r == [(0, (None, None), (None, None)), (0, (None, None), (None, None)),
             (0, (None, None), (None, None)), (0, (None, None), (None, None)),
             (2, (1, 0), (1, 1)), (3, (1, 0), (1, 0))]
r = test([[[-1, 5], [-2, 2]], [[-2, 4], [-4, 1]]])
assert r == [(2, (0, 0), (1, 1)), (3, (0, 1), (1, 1)),
             (3, (0, 0), (0, 1)), (4, (0, 1), (0, 1)),
             (3, (1, 0), (1, 1)), (5, (1, 1), (1, 1))]
r = test([[[-1, -1, -1, 2]], [[-1, 1, 1, 1]]])
assert r == [(2, (0, 0), (1, 0)), (2, (1, 0), (1, 0)),
             (3, (0, 0), (0, 0)), (3, (0, 0), (0, 0)),
             (2, (1, 0), (1, 0)), (2, (1, 0), (1, 0))]

