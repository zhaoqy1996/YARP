from __future__ import division
from math import pi

import numpy as np

from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, kpts2ndarray
from ase.units import Bohr, Ha


def make_test_dft_calculation():
    a = b = 2.0
    c = 6.0
    atoms = Atoms(positions=[(0, 0, c / 2)],
                  symbols='H',
                  pbc=(1, 1, 0),
                  cell=(a, b, c),
                  calculator=TestCalculator())
    return atoms


class TestCalculator:
    def __init__(self, nk=8):
        assert nk % 2 == 0
        bzk = []
        weights = []
        ibzk = []
        w = 1.0 / nk**2
        for i in range(-nk + 1, nk, 2):
            for j in range(-nk + 1, nk, 2):
                k = (0.5 * i / nk, 0.5 * j / nk, 0)
                bzk.append(k)
                if i >= j > 0:
                    ibzk.append(k)
                    if i == j:
                        weights.append(4 * w)
                    else:
                        weights.append(8 * w)
        assert abs(sum(weights) - 1.0) < 1e-12
        self.bzk = np.array(bzk)
        self.ibzk = np.array(ibzk)
        self.weights = np.array(weights)

        # Calculate eigenvalues and wave functions:
        self.init()

    def init(self):
        nibzk = len(self.weights)
        nbands = 1

        V = -1.0
        self.eps = 2 * V * (np.cos(2 * pi * self.ibzk[:, 0]) +
                            np.cos(2 * pi * self.ibzk[:, 1]))
        self.eps.shape = (nibzk, nbands)

        self.psi = np.zeros((nibzk, 20, 20, 60), complex)
        phi = np.empty((2, 2, 20, 20, 60))
        z = np.linspace(-1.5, 1.5, 60, endpoint=False)
        for i in range(2):
            x = np.linspace(0, 1, 20, endpoint=False) - i
            for j in range(2):
                y = np.linspace(0, 1, 20, endpoint=False) - j
                r = (((x[:, None]**2 +
                       y**2)[:, :, None] +
                      z**2)**0.5).clip(0, 1)
                phi = 1.0 - r**2 * (3.0 - 2.0 * r)
                phase = np.exp(pi * 2j * np.dot(self.ibzk, (i, j, 0)))
                self.psi += phase[:, None, None, None] * phi

    def get_pseudo_wave_function(self, band=0, kpt=0, spin=0):
        assert spin == 0 and band == 0
        return self.psi[kpt]

    def get_eigenvalues(self, kpt=0, spin=0):
        assert spin == 0
        return self.eps[kpt]

    def get_number_of_bands(self):
        return 1

    def get_k_point_weights(self):
        return self.weights

    def get_number_of_spins(self):
        return 1

    def get_fermi_level(self):
        return 0.0

    def get_pseudo_density(self):
        n = 0.0
        for w, eps, psi in zip(self.weights, self.eps[:, 0], self.psi):
            if eps >= 0.0:
                continue
            n += w * (psi * psi.conj()).real

        n[1:] += n[:0:-1].copy()
        n[:, 1:] += n[:, :0:-1].copy()
        n += n.transpose((1, 0, 2)).copy()
        n /= 8
        return n


class TestPotential(Calculator):
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        E = 0.0
        R = atoms.positions
        F = np.zeros_like(R)
        for a, r in enumerate(R):
            D = R - r
            d = (D**2).sum(1)**0.5
            x = d - 1.0
            E += np.vdot(x, x)
            d[a] = 1
            F -= (x / d)[:, None] * D
        energy = 0.25 * E
        self.results = {'energy': energy, 'forces': F}


class FreeElectrons(Calculator):
    """Free-electron band calculator.

    Parameters:

    nvalence: int
        Number of electrons
    kpts: dict
        K-point specification.

    Example:

    >>> calc = FreeElectrons(nvalence=1, kpts={'path': 'GXL'})
    """

    implemented_properties = ['energy']

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms)
        self.kpts = kpts2ndarray(self.parameters.kpts, atoms)
        icell = atoms.get_reciprocal_cell() * 2 * np.pi * Bohr
        n = 7
        offsets = np.indices((n, n, n)).T.reshape((n**3, 1, 3)) - n // 2
        eps = 0.5 * (np.dot(self.kpts + offsets, icell)**2).sum(2).T
        eps.sort()
        self.eigenvalues = eps[:, :20] * Ha
        self.results = {'energy': 0.0}

    def get_eigenvalues(self, kpt, spin=0):
        assert spin == 0
        return self.eigenvalues[kpt].copy()

    def get_fermi_level(self):
        v = self.atoms.get_volume() / Bohr**3
        kF = (self.parameters.nvalence / v * 3 * np.pi**2)**(1 / 3)
        return 0.5 * kF**2 * Ha

    def get_ibz_k_points(self):
        return self.kpts.copy()

    def get_number_of_spins(self):
        return 1


def numeric_force(atoms, a, i, d=0.001):
    """Compute numeric force on atom with index a, Cartesian component i,
    with finite step of size d
    """
    p0 = atoms.get_positions()
    p = p0.copy()
    p[a, i] += d
    atoms.set_positions(p, apply_constraint=False)
    eplus = atoms.get_potential_energy()
    p[a, i] -= 2 * d
    atoms.set_positions(p, apply_constraint=False)
    eminus = atoms.get_potential_energy()
    atoms.set_positions(p0, apply_constraint=False)
    return (eminus - eplus) / (2 * d)


def gradient_test(atoms, indices=None):
    """
    Use numeric_force to compare analytical and numerical forces on atoms

    If indices is None, test is done on all atoms.
    """
    if indices is None:
        indices = range(len(atoms))
    f = atoms.get_forces()[indices]
    print('{0:>16} {1:>20}'.format('eps', 'max(abs(df))'))
    for eps in np.logspace(-1, -8, 8):
        fn = np.zeros((len(indices), 3))
        for idx, i in enumerate(indices):
            for j in range(3):
                fn[idx, j] = numeric_force(atoms, i, j, eps)
        print('{0:16.12f} {1:20.12f}'.format(eps, abs(fn - f).max()))
    return f, fn
