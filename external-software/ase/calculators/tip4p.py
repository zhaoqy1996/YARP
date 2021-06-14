from __future__ import division
import numpy as np

import ase.units as unit
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.tip3p import rOH, angleHOH, TIP3P

__all__ = ['rOH', 'angleHOH', 'TIP4P', 'sigma0', 'epsilon0']

# Electrostatic constant and parameters:
k_c = 332.1 * unit.kcal / unit.mol
sigma0 = 3.15365
epsilon0 = 0.6480 * unit.kJ / unit.mol


class TIP4P(TIP3P):
    def __init__(self, rc=7.0, width=1.0):
        """ TIP4P potential for water.

        http://dx.doi.org/10.1063/1.445869

        Requires an atoms object of OHH,OHH, ... sequence
        Correct TIP4P charges and LJ parameters set automatically.

        Virtual interaction sites implemented in the following scheme:
        Original atoms object has no virtual sites.
        When energy/forces are requested:

        * virtual sites added to temporary xatoms object
        * energy / forces calculated
        * forces redistributed from virtual sites to actual atoms object

        This means you do not get into trouble when propagating your system
        with MD while having to skip / account for massless virtual sites.

        This also means that if using for QM/MM MD with GPAW, the EmbedTIP4P
        class must be used.
        """

        TIP3P.__init__(self, rc, width)
        self.energy = None
        self.forces = None

    def calculate(self, atoms=None,
                  properties=['energy', 'forces'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        assert (atoms.numbers[::3] == 8).all()
        assert (atoms.numbers[1::3] == 1).all()
        assert (atoms.numbers[2::3] == 1).all()

        xpos = self.add_virtual_sites(atoms.positions)
        xcharges = self.get_virtual_charges(atoms)

        cell = atoms.cell
        pbc = atoms.pbc

        natoms = len(atoms)
        nmol = natoms // 3

        self.energy = 0.0
        self.forces = np.zeros((4 * natoms // 3, 3))

        C = cell.diagonal()
        assert (cell == np.diag(C)).all(), 'not orthorhombic'
        assert ((C >= 2 * self.rc) | ~pbc).all(), 'cutoff too large'

        # Get dx,dy,dz from first atom of each mol to same atom of all other
        # and find min. distance. Everything moves according to this analysis.
        for a in range(nmol - 1):
            D = xpos[(a + 1) * 4::4] - xpos[a * 4]
            shift = np.zeros_like(D)
            for i, periodic in enumerate(pbc):
                if periodic:
                    shift[:, i] = np.rint(D[:, i] / C[i]) * C[i]
            q_v = xcharges[(a + 1) * 4:]

            # Min. img. position list as seen for molecule !a!
            position_list = np.zeros(((nmol - 1 - a) * 4, 3))

            for j in range(4):
                position_list[j::4] += xpos[(a + 1) * 4 + j::4] - shift

            # Make the smooth cutoff:
            pbcRoo = position_list[::4] - xpos[a * 4]
            pbcDoo = np.sum(np.abs(pbcRoo)**2, axis=-1)**(1 / 2)
            x1 = pbcDoo > self.rc - self.width
            x2 = pbcDoo < self.rc
            x12 = np.logical_and(x1, x2)
            y = (pbcDoo[x12] - self.rc + self.width) / self.width
            t = np.zeros(len(pbcDoo))
            t[x2] = 1.0
            t[x12] -= y**2 * (3.0 - 2.0 * y)
            dtdd = np.zeros(len(pbcDoo))
            dtdd[x12] -= 6.0 / self.width * y * (1.0 - y)
            self.energy_and_forces(a, xpos, position_list, q_v, nmol, t, dtdd)

        if self.pcpot:
            e, f = self.pcpot.calculate(xcharges, xpos)
            self.energy += e
            self.forces += f

        f = self.redistribute_forces(self.forces)

        self.results['energy'] = self.energy
        self.results['forces'] = f

    def energy_and_forces(self, a, xpos, position_list, q_v, nmol, t, dtdd):
        """ energy and forces on molecule a from all other molecules.
            cutoff is based on O-O Distance. """

        # LJ part - only O-O interactions
        epsil = np.tile([epsilon0], nmol - 1 - a)
        sigma = np.tile([sigma0], nmol - 1 - a)
        DOO = position_list[::4] - xpos[a * 4]
        d2 = (DOO**2).sum(1)
        d = np.sqrt(d2)
        e_lj = 4 * epsil * (sigma**12 / d**12 - sigma**6 / d**6)
        f_lj = (4 * epsil * (12 * sigma**12 / d**13 -
                             6 * sigma**6 / d**7) * t -
                e_lj * dtdd)[:, np.newaxis] * DOO / d[:, np.newaxis]

        self.forces[a * 4] -= f_lj.sum(0)
        self.forces[(a + 1) * 4::4] += f_lj

        # Electrostatics
        e_elec = 0
        all_cut = np.repeat(t, 4)
        for i in range(4):
            D = position_list - xpos[a * 4 + i]
            d2_all = (D**2).sum(axis=1)
            d_all = np.sqrt(d2_all)
            e = k_c * q_v[i] * q_v / d_all
            e_elec += np.dot(all_cut, e).sum()
            e_f = e.reshape(nmol - a - 1, 4).sum(1)
            F = (e / d_all * all_cut)[:, np.newaxis] * D / d_all[:, np.newaxis]
            FOO = -(e_f * dtdd)[:, np.newaxis] * DOO / d[:, np.newaxis]
            self.forces[(a + 1) * 4 + 0::4] += FOO
            self.forces[a * 4] -= FOO.sum(0)
            self.forces[(a + 1) * 4:] += F
            self.forces[a * 4 + i] -= F.sum(0)

        self.energy += np.dot(e_lj, t) + e_elec

    def add_virtual_sites(self, pos):
        # Order: OHHM,OHHM,...
        # DOI: 10.1002/(SICI)1096-987X(199906)20:8
        b = 0.15
        xatomspos = np.zeros((4 * len(pos) // 3, 3))
        for w in range(0, len(pos), 3):
            r_i = pos[w]  # O pos
            r_j = pos[w + 1]  # H1 pos
            r_k = pos[w + 2]  # H2 pos
            n = (r_j + r_k) / 2 - r_i
            n /= np.linalg.norm(n)
            r_d = r_i + b * n

            x = 4 * w // 3
            xatomspos[x + 0] = r_i
            xatomspos[x + 1] = r_j
            xatomspos[x + 2] = r_k
            xatomspos[x + 3] = r_d

        return xatomspos

    def get_virtual_charges(self, atoms):
        charges = np.empty(len(atoms) * 4 // 3)
        charges[0::4] = 0.00  # O
        charges[1::4] = 0.52  # H1
        charges[2::4] = 0.52  # H2
        charges[3::4] = -1.04  # X1
        return charges

    def redistribute_forces(self, forces):
        f = forces
        b = 0.15
        a = 0.5
        pos = self.atoms.positions
        for w in range(0, len(pos), 3):
            r_i = pos[w]  # O pos
            r_j = pos[w + 1]  # H1 pos
            r_k = pos[w + 2]  # H2 pos
            r_ij = r_j - r_i
            r_jk = r_k - r_j
            r_d = r_i + b * (r_ij + a * r_jk) / np.linalg.norm(r_ij + a * r_jk)
            r_id = r_d - r_i
            gamma = b / np.linalg.norm(r_ij + a * r_jk)

            x = w * 4 // 3
            Fd = f[x + 3]  # force on M
            F1 = (np.dot(r_id, Fd) / np.dot(r_id, r_id)) * r_id
            Fi = Fd - gamma * (Fd - F1)  # Force from M on O
            Fj = (1 - a) * gamma * (Fd - F1)  # Force from M on H1
            Fk = a * gamma * (Fd - F1)  # Force from M on H2

            f[x] += Fi
            f[x + 1] += Fj
            f[x + 2] += Fk

        # remove virtual sites from force array
        f = np.delete(f, list(range(3, f.shape[0], 4)), axis=0)
        return f
