import functools
from math import pi, sqrt

import numpy as np

from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.parallel import world


class DOS:
    def __init__(self, calc, width=0.1, window=None, npts=401):
        """Electronic Density Of States object.

        calc: calculator object
            Any ASE compliant calculator object.
        width: float
            Width of guassian smearing.  Use width=0.0 for linear tetrahedron
            interpolation.
        window: tuple of two float
            Use ``window=(emin, emax)``.  If not specified, a window
            big enough to hold all the eigenvalues will be used.
        npts: int
            Number of points.

        """

        self.npts = npts
        self.width = width
        self.w_k = calc.get_k_point_weights()
        self.nspins = calc.get_number_of_spins()
        self.e_skn = np.array([[calc.get_eigenvalues(kpt=k, spin=s)
                                for k in range(len(self.w_k))]
                               for s in range(self.nspins)])
        self.e_skn -= calc.get_fermi_level()

        if window is None:
            emin = None
            emax = None
        else:
            emin, emax = window

        if emin is None:
            emin = self.e_skn.min() - 5 * self.width
        if emax is None:
            emax = self.e_skn.max() + 5 * self.width

        self.energies = np.linspace(emin, emax, npts)

        if width == 0.0:
            bzkpts = calc.get_bz_k_points()
            size, offset = get_monkhorst_pack_size_and_offset(bzkpts)
            bz2ibz = calc.get_bz_to_ibz_map()
            shape = (self.nspins,) + tuple(size) + (-1,)
            self.e_skn = self.e_skn[:, bz2ibz].reshape(shape)
            self.cell = calc.atoms.cell

    def get_energies(self):
        """Return the array of energies used to sample the DOS.

        The energies are reported relative to the Fermi level.
        """
        return self.energies

    def delta(self, energy):
        """Return a delta-function centered at 'energy'."""
        x = -((self.energies - energy) / self.width)**2
        return np.exp(x) / (sqrt(pi) * self.width)

    def get_dos(self, spin=None):
        """Get array of DOS values.

        The *spin* argument can be 0 or 1 (spin up or down) - if not
        specified, the total DOS is returned.
        """

        if spin is None:
            if self.nspins == 2:
                # Spin-polarized calculation, but no spin specified -
                # return the total DOS:
                return self.get_dos(spin=0) + self.get_dos(spin=1)
            else:
                spin = 0

        if self.width == 0.0:
            return ltidos(self.cell, self.e_skn[spin], self.energies)

        dos = np.zeros(self.npts)
        for w, e_n in zip(self.w_k, self.e_skn[spin]):
            for e in e_n:
                dos += w * self.delta(e)
        return dos


def ltidos(cell, eigs, energies, weights=None):
    """DOS from linear tetrahedron interpolation.

    cell: 3x3 ndarray-like
        Unit cell.
    eigs: (n1, n2, n3, nbands)-shaped ndarray
        Eigenvalues on a Monkhorst-Pack grid (not reduced).
    energies: 1-d array-like
        Energies where the DOS is calculated (must be a uniform grid).
    weights: (n1, n2, n3, nbands)-shaped ndarray
        Weights.  Defaults to 1.
    """

    from scipy.spatial import Delaunay

    I, J, K = size = eigs.shape[:3]
    B = (np.linalg.inv(cell) / size).T
    indices = np.array([[i, j, k]
                        for i in [0, 1] for j in [0, 1] for k in [0, 1]])
    dt = Delaunay(np.dot(indices, B))

    dos = np.zeros_like(energies)
    integrate = functools.partial(_lti, energies, dos)

    for s in dt.simplices:
        kpts = dt.points[s]
        try:
            M = np.linalg.inv(kpts[1:, :] - kpts[0, :])
        except np.linalg.linalg.LinAlgError:
            continue
        n = -1
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    n += 1
                    if n % world.size != world.rank:
                        continue
                    E = np.array([eigs[(i + a) % I, (j + b) % J, (k + c) % K]
                                  for a, b, c in indices[s]])
                    if weights is None:
                        integrate(kpts, M, E)
                    else:
                        w = np.array([weights[(i + a) % I, (j + b) % J,
                                              (k + c) % K]
                                      for a, b, c in indices[s]])
                        integrate(kpts, M, E, w)

    world.sum(dos)

    return dos * abs(np.linalg.det(cell))


def _lti(energies, dos, kpts, M, E, W=None):
    zero = energies[0]
    de = energies[1] - zero
    for z, e in enumerate(E.T):
        dedk = (np.dot(M, e[1:] - e[0])**2).sum()**0.5
        i = e.argsort()
        k = kpts[i, :, np.newaxis]
        e0, e1, e2, e3 = ee = e[i]
        for j in range(3):
            m = max(0, int((ee[j] - zero) / de) + 1)
            n = min(len(energies) - 1, int((ee[j + 1] - zero) / de) + 1)
            if n > m:
                v = energies[m:n]
                if j == 0:
                    x10 = (e1 - v) / (e1 - e0)
                    x01 = (v - e0) / (e1 - e0)
                    x20 = (e2 - v) / (e2 - e0)
                    x02 = (v - e0) / (e2 - e0)
                    x30 = (e3 - v) / (e3 - e0)
                    x03 = (v - e0) / (e3 - e0)
                    k1 = k[0] * x10 + k[1] * x01
                    k2 = k[0] * x20 + k[2] * x02 - k1
                    k3 = k[0] * x30 + k[3] * x03 - k1
                    if W is None:
                        w = 0.5 / dedk
                    else:
                        w = np.dot(W[i, z],
                                   [x10 + x20 + x30, x01, x02, x03])
                        w /= 6 * dedk
                    dos[m:n] += (np.cross(k2, k3, 0, 0)**2).sum(1)**0.5 * w
                elif j == 1:
                    x21 = (e2 - v) / (e2 - e1)
                    x12 = (v - e1) / (e2 - e1)
                    x20 = (e2 - v) / (e2 - e0)
                    x02 = (v - e0) / (e2 - e0)
                    x30 = (e3 - v) / (e3 - e0)
                    x03 = (v - e0) / (e3 - e0)
                    x31 = (e3 - v) / (e3 - e1)
                    x13 = (v - e1) / (e3 - e1)
                    k1 = k[1] * x21 + k[2] * x12
                    k2 = k[0] * x20 + k[2] * x02 - k1
                    k3 = k[0] * x30 + k[3] * x03 - k1
                    k4 = k[1] * x31 + k[3] * x13 - k1
                    if W is None:
                        w = 0.5 / dedk
                    else:
                        w = np.dot(W[i, z],
                                   [x20 + x30, x21 + x31,
                                    x12 + x02, x03 + x13])
                        w /= 8 * dedk
                    dos[m:n] += (np.cross(k2, k3, 0, 0)**2).sum(1)**0.5 * w
                    dos[m:n] += (np.cross(k4, k3, 0, 0)**2).sum(1)**0.5 * w
                elif j == 2:
                    x30 = (e3 - v) / (e3 - e0)
                    x03 = (v - e0) / (e3 - e0)
                    x31 = (e3 - v) / (e3 - e1)
                    x13 = (v - e1) / (e3 - e1)
                    x32 = (e3 - v) / (e3 - e2)
                    x23 = (v - e2) / (e3 - e2)
                    k1 = k[0] * x30 + k[3] * x03
                    k2 = k[1] * x31 + k[3] * x13 - k1
                    k3 = k[2] * x32 + k[3] * x23 - k1
                    if W is None:
                        w = 0.5 / dedk
                    else:
                        w = np.dot(W[i, z],
                                   [x30, x31, x32, x03 + x13 + x23])
                        w /= 6 * dedk
                    dos[m:n] += (np.cross(k2, k3, 0, 0)**2).sum(1)**0.5 * w
