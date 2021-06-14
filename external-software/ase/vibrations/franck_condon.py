from __future__ import division
from functools import reduce
from itertools import combinations, chain
from math import factorial
from operator import mul

import numpy as np

from ase.units import kg, C, _hbar, kB
from ase.vibrations import Vibrations


class Factorial:
    def __init__(self):
        self._fac = [1]
        self._inv = [1.]

    def __call__(self, n):
        try:
            return self._fac[n]
        except IndexError:
            for i in range(len(self._fac), n + 1):
                self._fac.append(i * self._fac[i - 1])
                try:
                    self._inv.append(float(1. / self._fac[-1]))
                except OverflowError:
                    self._inv.append(0.)
            return self._fac[n]

    def inv(self, n):
        self(n)
        return self._inv[n]


class FranckCondonOverlap:
    """Evaluate squared overlaps depending on the Huang-Rhys parameter."""
    def __init__(self):
        self.factorial = Factorial()

    def directT0(self, n, S):
        """|<0|n>|^2

        Direct squared Franck-Condon overlap corresponding to T=0.
        """
        return np.exp(-S) * S**n * self.factorial.inv(n)

    def direct(self, n, m, S_in):
        """|<n|m>|^2

        Direct squared Franck-Condon overlap.
        """
        if n > m:
            # use symmetry
            return self.direct(m, n, S_in)

        S = np.array([S_in])
        mask = np.where(S == 0)
        S[mask] = 1  # hide zeros
        s = 0
        for k in range(n + 1):
            s += (-1)**(n - k) * S**float(-k) / (
                self.factorial(k) *
                self.factorial(n - k) * self.factorial(m - k))
        res = np.exp(-S) * S**(n + m) * s**2 * (
            self.factorial(n) * self.factorial(m))
        # use othogonality
        res[mask] = int(n == m)
        return res[0]

    def direct0mm1(self, m, S):
        """<0|m><m|1>"""
        sum = S**m
        if m:
            sum -= m * S**(m - 1)
        return np.exp(-S) * np.sqrt(S) * sum * self.factorial.inv(m)

    def direct0mm2(self, m, S):
        """<0|m><m|2>"""
        sum = S**(m + 1)
        if m >= 1:
            sum -= 2 * m * S**m
        if m >= 2:
            sum += m * (m - 1) * S**(m - 1)
        return np.exp(-S) / np.sqrt(2) * sum * self.factorial.inv(m)


class FranckCondonRecursive:
    """Recursive implementation of Franck-Condon overlaps

    Notes
    -----
    The ovelaps are signed according to the sign of the displacements.

    Reference
    ---------
    Julien Guthmuller
    The Journal of Chemical Physics 144, 064106 (2016); doi: 10.1063/1.4941449
    """
    def __init__(self):
        self.factorial = Factorial()

    def ov0m(self, m, delta):
        if m == 0:
            return np.exp(-0.25 * delta**2)
        else:
            assert(m > 0)
            return - delta / np.sqrt(2 * m) * self.ov0m(m - 1, delta)
            
    def ov1m(self, m, delta):
        sum = delta * self.ov0m(m, delta) / np.sqrt(2.)
        if m == 0:
            return sum
        else:
            assert(m > 0)
            return sum + np.sqrt(m) * self.ov0m(m - 1, delta)
            
    def ov2m(self, m, delta):
        sum = delta * self.ov1m(m, delta) / 2
        if m == 0:
            return sum
        else:
            assert(m > 0)
            return sum + np.sqrt(m / 2.) * self.ov1m(m - 1, delta)
            
    def ov3m(self, m, delta):
        sum = delta * self.ov2m(m, delta) / np.sqrt(6.)
        if m == 0:
            return sum
        else:
            assert(m > 0)
            return sum + np.sqrt(m / 3.) * self.ov2m(m - 1, delta)
            
    def ov0mm1(self, m, delta):
        if m == 0:
            return delta / np.sqrt(2) * self.ov0m(m, delta)**2
        else:
            return delta / np.sqrt(2) * (
                self.ov0m(m, delta)**2 - self.ov0m(m - 1, delta)**2)
            
    def direct0mm1(self, m, delta):
        """direct and fast <0|m><m|1>"""
        S = delta**2 / 2.
        sum = S**m
        if m:
            sum -= m * S**(m - 1)
        return np.where(S == 0, 0,
                        (np.exp(-S) * delta / np.sqrt(2) * sum *
                         self.factorial.inv(m)))

    def ov0mm2(self, m, delta):
        if m == 0:
            return delta**2 / np.sqrt(8) * self.ov0m(m, delta)**2
        elif m == 1:
            return delta**2 / np.sqrt(8) * (
                self.ov0m(m, delta)**2 - 2 * self.ov0m(m - 1, delta)**2)
        else:
            return delta**2 / np.sqrt(8) * (
                self.ov0m(m, delta)**2 - 2 * self.ov0m(m - 1, delta)**2 +
                self.ov0m(m - 2, delta)**2)

    def direct0mm2(self, m, delta):
        """direct and fast <0|m><m|2>"""
        S = delta**2 / 2.
        sum = S**(m + 1)
        if m >= 1:
            sum -= 2 * m * S**m
        if m >= 2:
            sum += m * (m - 1) * S**(m - 1)
        return np.exp(-S) / np.sqrt(2) * sum * self.factorial.inv(m)

    def ov1mm2(self, m, delta):
        p1 = delta**3 / 4.
        sum = p1 * self.ov0m(m, delta)**2
        if m == 0:
            return sum
        p2 = delta - 3. * delta**3 / 4
        sum += p2 * self.ov0m(m - 1, delta)**2
        if m == 1:
            return sum
        sum -= p2 * self.ov0m(m - 2, delta)**2
        if m == 2:
            return sum
        return sum - p1 * self.ov0m(m - 3, delta)**2

    def direct1mm2(self, m, delta):
        S = delta**2 / 2.
        sum = S**2
        if m > 0:
            sum -= 2 * m * S
        if m > 1:
            sum += m * (m - 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(S == 0, 0,
                (np.exp(-S) * S**(m - 1) / delta * (S - m) * sum *
                 self.factorial.inv(m)))

    def direct0mm3(self, m, delta):
        S = delta**2 / 2.
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(S == 0, 0,
                (np.exp(-S) * S**(m - 1) / delta * np.sqrt(12.) *
                 (S**3 / 6. - m * S**2 / 2 +
                  m * (m - 1) * S / 2. - m * (m - 1) * (m - 2) / 6) *
                 self.factorial.inv(m)))


class FranckCondon:
    def __init__(self, atoms, vibname, minfreq=-np.inf, maxfreq=np.inf):
        """Input is a atoms object and the corresponding vibrations.
        With minfreq and maxfreq frequencies can
        be excluded from the calculation"""

        self.atoms = atoms
        # V = a * v is the combined atom and xyz-index
        self.mm05_V = np.repeat(1. / np.sqrt(atoms.get_masses()), 3)
        self.minfreq = minfreq
        self.maxfreq = maxfreq
        self.shape = (len(self.atoms), 3)

        vib = Vibrations(atoms, name=vibname)
        self.energies = np.real(vib.get_energies(method='frederiksen'))  # [eV]
        self.frequencies = np.real(
            vib.get_frequencies(method='frederiksen'))  # [cm^-1]
        self.modes = vib.modes
        self.H = vib.H

    def get_Huang_Rhys_factors(self, forces):
        """Evaluate Huang-Rhys factors and corresponding frequencies
        from forces on atoms in the exited electronic state.
        The double harmonic approximation is used. HR factors are
        the first approximation of FC factors,
        no combinations or higher quanta (>1) exitations are considered"""

        assert(forces.shape == self.shape)

        # Hesse matrix
        H_VV = self.H
        # sqrt of inverse mass matrix
        mm05_V = self.mm05_V
        # mass weighted Hesse matrix
        Hm_VV = mm05_V[:, None] * H_VV * mm05_V
        # mass weighted displacements
        Fm_V = forces.flat * mm05_V
        X_V = np.linalg.solve(Hm_VV, Fm_V)
        # projection onto the modes
        modes_VV = self.modes
        d_V = np.dot(modes_VV, X_V)
        # Huang-Rhys factors S
        s = 1.e-20 / kg / C / _hbar**2  # SI units
        S_V = s * d_V**2 * self.energies / 2

        # reshape for minfreq
        indices = np.where(self.frequencies <= self.minfreq)
        np.append(indices, np.where(self.frequencies >= self.maxfreq))
        S_V = np.delete(S_V, indices)
        frequencies = np.delete(self.frequencies, indices)

        return S_V, frequencies

    def get_Franck_Condon_factors(self, order, temp, forces):
        """Return FC factors and corresponding frequencies up to given order.

        order= number of quanta taken into account
        T= temperature in K. Vibronic levels are occupied by a
        Boltzman distribution.
        forces= forces on atoms in the exited electronic state"""

        S, f = self.get_Huang_Rhys_factors(forces)
        n = order + 1
        T = temp
        freq = np.array(f)

        # frequencies
        freq_n = [[] * i for i in range(n - 1)]
        freq_neg = [[] * i for i in range(n - 1)]

        for i in range(1, n):
            freq_n[i - 1] = freq * i
            freq_neg[i - 1] = freq * (-i)

        # combinations
        freq_nn = [x for x in combinations(chain(*freq_n), 2)]
        for i in range(len(freq_nn)):
            freq_nn[i] = freq_nn[i][0] + freq_nn[i][1]

        indices2 = []
        for i, y in enumerate(freq):
            ind = [j for j, x in enumerate(freq_nn) if x % y == 0]
            indices2.append(ind)
        indices2 = [x for x in chain(*indices2)]
        freq_nn = np.delete(freq_nn, indices2)

        frequencies = [[] * x for x in range(3)]
        frequencies[0].append(freq_neg[0])
        frequencies[0].append([0])
        frequencies[0].append(freq_n[0])
        frequencies[0] = [x for x in chain(*frequencies[0])]

        for i in range(1, n - 1):
            frequencies[1].append(freq_neg[i])
            frequencies[1].append(freq_n[i])
        frequencies[1] = [x for x in chain(*frequencies[1])]

        frequencies[2] = freq_nn

        # Franck-Condon factors
        E = freq / 8065.5
        f_n = [[] * i for i in range(n)]

        for j in range(0, n):
            f_n[j] = np.exp(-E * j / (kB * T))

        # partition function
        Z = np.empty(len(S))
        Z = np.sum(f_n, 0)

        # occupation probability
        w_n = [[] * k for k in range(n)]
        for l in range(n):
            w_n[l] = f_n[l] / Z

        # overlap wavefunctions
        O_n = [[] * m for m in range(n)]
        O_neg = [[] * m for m in range(n)]
        for o in range(n):
            O_n[o] = [[] * p for p in range(n)]
            O_neg[o] = [[] * p for p in range(n - 1)]
            for q in range(o, n + o):
                a = np.minimum(o, q)
                summe = []
                for k in range(a + 1):
                    s = ((-1)**(q - k) * np.sqrt(S)**(o + q - 2 * k) *
                         factorial(o) * factorial(q) /
                         (factorial(k) * factorial(o - k) * factorial(q - k)))
                    summe.append(s)
                summe = np.sum(summe, 0)
                O_n[o][q - o] = (np.exp(-S / 2) /
                                 (factorial(o) * factorial(q))**(0.5) *
                                 summe)**2 * w_n[o]
            for q in range(n - 1):
                O_neg[o][q] = [0 * b for b in range(len(S))]
            for q in range(o - 1, -1, -1):
                a = np.minimum(o, q)
                summe = []
                for k in range(a + 1):
                    s = ((-1)**(q - k) * np.sqrt(S)**(o + q - 2 * k) *
                         factorial(o) * factorial(q) /
                         (factorial(k) * factorial(o - k) * factorial(q - k)))
                    summe.append(s)
                summe = np.sum(summe, 0)
                O_neg[o][q] = (np.exp(-S / 2) /
                               (factorial(o) * factorial(q))**(0.5) *
                               summe)**2 * w_n[o]
        O_neg = np.delete(O_neg, 0, 0)

        # Franck-Condon factors
        FC_n = [[] * i for i in range(n)]
        FC_n = np.sum(O_n, 0)
        zero = reduce(mul, FC_n[0])
        FC_neg = [[] * i for i in range(n - 2)]
        FC_neg = np.sum(O_neg, 0)
        FC_n = np.delete(FC_n, 0, 0)

        # combination FC factors
        FC_nn = [x for x in combinations(chain(*FC_n), 2)]
        for i in range(len(FC_nn)):
            FC_nn[i] = FC_nn[i][0] * FC_nn[i][1]

        FC_nn = np.delete(FC_nn, indices2)

        FC = [[] * x for x in range(3)]
        FC[0].append(FC_neg[0])
        FC[0].append([zero])
        FC[0].append(FC_n[0])
        FC[0] = [x for x in chain(*FC[0])]

        for i in range(1, n - 1):
            FC[1].append(FC_neg[i])
            FC[1].append(FC_n[i])
        FC[1] = [x for x in chain(*FC[1])]

        FC[2] = FC_nn

        """Returned are two 3-dimensional lists. First inner list contains
frequencies and FC-factors of vibrations exited with |1| quanta and
the 0-0 transition.
        Second list contains frequencies and FC-factors from higher
quanta exitations. Third list are combinations of two normal modes
(including combinations of higher quanta exitations). """
        return FC, frequencies
