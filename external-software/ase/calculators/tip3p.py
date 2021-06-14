"""TIP3P potential."""
from __future__ import division

import numpy as np

import ase.units as units
from ase.calculators.calculator import Calculator, all_changes

qH = 0.417
sigma0 = 3.15061
epsilon0 = 0.1521 * units.kcal / units.mol
rOH = 0.9572
angleHOH = 104.52
thetaHOH = 104.52 / 180 * np.pi  # we keep this for backwards compatibility


class TIP3P(Calculator):
    implemented_properties = ['energy', 'forces']
    nolabel = True
    pcpot = None

    def __init__(self, rc=5.0, width=1.0):
        """TIP3P potential.

        rc: float
            Cutoff radius for Coulomb part.
        width: float
            Width for cutoff function for Coulomb part.
        """
        self.rc = rc
        self.width = width
        Calculator.__init__(self)

    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        R = self.atoms.positions.reshape((-1, 3, 3))
        Z = self.atoms.numbers
        pbc = self.atoms.pbc
        cell = self.atoms.cell.diagonal()
        nh2o = len(R)

        assert (self.atoms.cell == np.diag(cell)).all(), 'not orthorhombic'
        assert ((cell >= 2 * self.rc) | ~pbc).all(), 'cutoff too large'  # ???
        if Z[0] == 8:
            o = 0
        else:
            o = 2
        assert (Z[o::3] == 8).all()
        assert (Z[(o + 1) % 3::3] == 1).all()
        assert (Z[(o + 2) % 3::3] == 1).all()

        charges = np.array([qH, qH, qH])
        charges[o] *= -2

        energy = 0.0
        forces = np.zeros((3 * nh2o, 3))

        for m in range(nh2o - 1):
            DOO = R[m + 1:, o] - R[m, o]
            shift = np.zeros_like(DOO)
            for i, periodic in enumerate(pbc):
                if periodic:
                    L = cell[i]
                    shift[:, i] = (DOO[:, i] + L / 2) % L - L / 2 - DOO[:, i]
            DOO += shift
            d2 = (DOO**2).sum(1)
            d = d2**0.5
            x1 = d > self.rc - self.width
            x2 = d < self.rc
            x12 = np.logical_and(x1, x2)
            y = (d[x12] - self.rc + self.width) / self.width
            t = np.zeros(len(d))  # cutoff function
            t[x2] = 1.0
            t[x12] -= y**2 * (3.0 - 2.0 * y)
            dtdd = np.zeros(len(d))
            dtdd[x12] -= 6.0 / self.width * y * (1.0 - y)
            c6 = (sigma0**2 / d2)**3
            c12 = c6**2
            e = 4 * epsilon0 * (c12 - c6)
            energy += np.dot(t, e)
            F = (24 * epsilon0 * (2 * c12 - c6) / d2 * t -
                 e * dtdd / d)[:, np.newaxis] * DOO
            forces[m * 3 + o] -= F.sum(0)
            forces[m * 3 + 3 + o::3] += F

            for j in range(3):
                D = R[m + 1:] - R[m, j] + shift[:, np.newaxis]
                r2 = (D**2).sum(axis=2)
                r = r2**0.5
                e = charges[j] * charges / r * units.Hartree * units.Bohr
                energy += np.dot(t, e).sum()
                F = (e / r2 * t[:, np.newaxis])[:, :, np.newaxis] * D
                FOO = -(e.sum(1) * dtdd / d)[:, np.newaxis] * DOO
                forces[(m + 1) * 3 + o::3] += FOO
                forces[m * 3 + o] -= FOO.sum(0)
                forces[(m + 1) * 3:] += F.reshape((-1, 3))
                forces[m * 3 + j] -= F.sum(axis=0).sum(axis=0)

        if self.pcpot:
            e, f = self.pcpot.calculate(np.tile(charges, nh2o),
                                        self.atoms.positions)
            energy += e
            forces += f

        self.results['energy'] = energy
        self.results['forces'] = forces

    def embed(self, charges):
        """Embed atoms in point-charges."""
        self.pcpot = PointChargePotential(charges)
        return self.pcpot

    def check_state(self, atoms, tol=1e-15):
        system_changes = Calculator.check_state(self, atoms, tol)
        if self.pcpot and self.pcpot.mmpositions is not None:
            system_changes.append('positions')
        return system_changes

    def add_virtual_sites(self, positions):
        return positions  # no virtual sites

    def redistribute_forces(self, forces):
        return forces

    def get_virtual_charges(self, atoms):
        charges = np.empty(len(atoms))
        charges[:] = qH
        if atoms.numbers[0] == 8:
            charges[::3] = -2 * qH
        else:
            charges[2::3] = -2 * qH
        return charges


class PointChargePotential:
    def __init__(self, mmcharges):
        """Point-charge potential for TIP3P.

        Only used for testing QMMM.
        """
        self.mmcharges = mmcharges
        self.mmpositions = None
        self.mmforces = None

    def set_positions(self, mmpositions, com_pv=None):
        self.mmpositions = mmpositions

    def calculate(self, qmcharges, qmpositions):
        energy = 0.0
        self.mmforces = np.zeros_like(self.mmpositions)
        qmforces = np.zeros_like(qmpositions)
        for C, R, F in zip(self.mmcharges, self.mmpositions, self.mmforces):
            d = qmpositions - R
            r2 = (d**2).sum(1)
            e = units.Hartree * units.Bohr * C * r2**-0.5 * qmcharges
            energy += e.sum()
            f = (e / r2)[:, np.newaxis] * d
            qmforces += f
            F -= f.sum(0)
        self.mmpositions = None
        return energy, qmforces

    def get_forces(self, calc):
        return self.mmforces
