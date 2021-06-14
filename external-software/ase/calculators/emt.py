"""Effective medium theory potential."""

from math import sqrt, exp, log

import numpy as np

from ase.data import chemical_symbols, atomic_numbers
from ase.units import Bohr
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes


parameters = {
    #      E0     s0    V0     eta2    kappa   lambda  n0
    #      eV     bohr  eV     bohr^-1 bohr^-1 bohr^-1 bohr^-3
    'Al': (-3.28, 3.00, 1.493, 1.240, 2.000, 1.169, 0.00700),
    'Cu': (-3.51, 2.67, 2.476, 1.652, 2.740, 1.906, 0.00910),
    'Ag': (-2.96, 3.01, 2.132, 1.652, 2.790, 1.892, 0.00547),
    'Au': (-3.80, 3.00, 2.321, 1.674, 2.873, 2.182, 0.00703),
    'Ni': (-4.44, 2.60, 3.673, 1.669, 2.757, 1.948, 0.01030),
    'Pd': (-3.90, 2.87, 2.773, 1.818, 3.107, 2.155, 0.00688),
    'Pt': (-5.85, 2.90, 4.067, 1.812, 3.145, 2.192, 0.00802),
    # extra parameters - just for fun ...
    'H': (-3.21, 1.31, 0.132, 2.652, 2.790, 3.892, 0.00547),
    'C': (-3.50, 1.81, 0.332, 1.652, 2.790, 1.892, 0.01322),
    'N': (-5.10, 1.88, 0.132, 1.652, 2.790, 1.892, 0.01222),
    'O': (-4.60, 1.95, 0.332, 1.652, 2.790, 1.892, 0.00850)}

beta = 1.809  # (16 * pi / 3)**(1.0 / 3) / 2**0.5, preserve historical rounding


class EMT(Calculator):
    """Python implementation of the Effective Medium Potential.

    Supports the following standard EMT metals:
    Al, Cu, Ag, Au, Ni, Pd and Pt.

    In addition, the following elements are supported.
    They are NOT well described by EMT, and the parameters
    are not for any serious use:
    H, C, N, O

    The potential takes a single argument, ``asap_cutoff``
    (default: False).  If set to True, the cutoff mimics
    how Asap does it; most importantly the global cutoff
    is chosen from the largest atom present in the simulation,
    if False it is chosen from the largest atom in the parameter
    table.  True gives the behaviour of the Asap code and
    older EMT implementations, although the results are not
    bitwise identical.
    """
    implemented_properties = ['energy', 'forces']

    nolabel = True

    default_parameters = {'asap_cutoff': False}

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def initialize(self, atoms):
        self.par = {}
        self.rc = 0.0
        self.numbers = atoms.get_atomic_numbers()
        if self.parameters.asap_cutoff:
            relevant_pars = {}
            for symb, p in parameters.items():
                if atomic_numbers[symb] in self.numbers:
                    relevant_pars[symb] = p
        else:
            relevant_pars = parameters
        maxseq = max(par[1] for par in relevant_pars.values()) * Bohr
        rc = self.rc = beta * maxseq * 0.5 * (sqrt(3) + sqrt(4))
        rr = rc * 2 * sqrt(4) / (sqrt(3) + sqrt(4))
        self.acut = np.log(9999.0) / (rr - rc)
        if self.parameters.asap_cutoff:
            self.rc_list = self.rc * 1.045
        else:
            self.rc_list = self.rc + 0.5
        for Z in self.numbers:
            if Z not in self.par:
                sym = chemical_symbols[Z]
                if sym not in parameters:
                    raise NotImplementedError('No EMT-potential for {0}'
                                              .format(sym))
                p = parameters[sym]
                s0 = p[1] * Bohr
                eta2 = p[3] / Bohr
                kappa = p[4] / Bohr
                x = eta2 * beta * s0
                gamma1 = 0.0
                gamma2 = 0.0
                for i, n in enumerate([12, 6, 24]):
                    r = s0 * beta * sqrt(i + 1)
                    x = n / (12 * (1.0 + exp(self.acut * (r - rc))))
                    gamma1 += x * exp(-eta2 * (r - beta * s0))
                    gamma2 += x * exp(-kappa / beta * (r - beta * s0))

                self.par[Z] = {'E0': p[0],
                               's0': s0,
                               'V0': p[2],
                               'eta2': eta2,
                               'kappa': kappa,
                               'lambda': p[5] / Bohr,
                               'n0': p[6] / Bohr**3,
                               'rc': rc,
                               'gamma1': gamma1,
                               'gamma2': gamma2}

        self.ksi = {}
        for s1, p1 in self.par.items():
            self.ksi[s1] = {}
            for s2, p2 in self.par.items():
                self.ksi[s1][s2] = p2['n0'] / p1['n0']

        self.forces = np.empty((len(atoms), 3))
        self.sigma1 = np.empty(len(atoms))
        self.deds = np.empty(len(atoms))

        self.nl = NeighborList([0.5 * self.rc_list] * len(atoms),
                               self_interaction=False)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if 'numbers' in system_changes:
            self.initialize(self.atoms)

        positions = self.atoms.positions
        numbers = self.atoms.numbers
        cell = self.atoms.cell

        self.nl.update(self.atoms)

        self.energy = 0.0
        self.sigma1[:] = 0.0
        self.forces[:] = 0.0

        natoms = len(self.atoms)

        for a1 in range(natoms):
            Z1 = numbers[a1]
            p1 = self.par[Z1]
            ksi = self.ksi[Z1]
            neighbors, offsets = self.nl.get_neighbors(a1)
            offsets = np.dot(offsets, cell)
            for a2, offset in zip(neighbors, offsets):
                d = positions[a2] + offset - positions[a1]
                r = sqrt(np.dot(d, d))
                if r < self.rc_list:
                    Z2 = numbers[a2]
                    p2 = self.par[Z2]
                    self.interact1(a1, a2, d, r, p1, p2, ksi[Z2])

        for a in range(natoms):
            Z = numbers[a]
            p = self.par[Z]
            try:
                ds = -log(self.sigma1[a] / 12) / (beta * p['eta2'])
            except (OverflowError, ValueError):
                self.deds[a] = 0.0
                self.energy -= p['E0']
                continue
            x = p['lambda'] * ds
            y = exp(-x)
            z = 6 * p['V0'] * exp(-p['kappa'] * ds)
            self.deds[a] = ((x * y * p['E0'] * p['lambda'] + p['kappa'] * z) /
                            (self.sigma1[a] * beta * p['eta2']))
            self.energy += p['E0'] * ((1 + x) * y - 1) + z

        for a1 in range(natoms):
            Z1 = numbers[a1]
            p1 = self.par[Z1]
            ksi = self.ksi[Z1]
            neighbors, offsets = self.nl.get_neighbors(a1)
            offsets = np.dot(offsets, cell)
            for a2, offset in zip(neighbors, offsets):
                d = positions[a2] + offset - positions[a1]
                r = sqrt(np.dot(d, d))
                if r < self.rc_list:
                    Z2 = numbers[a2]
                    p2 = self.par[Z2]
                    self.interact2(a1, a2, d, r, p1, p2, ksi[Z2])

        self.results['energy'] = self.energy
        self.results['free_energy'] = self.energy
        self.results['forces'] = self.forces

    def interact1(self, a1, a2, d, r, p1, p2, ksi):
        x = exp(self.acut * (r - self.rc))
        theta = 1.0 / (1.0 + x)
        y1 = (0.5 * p1['V0'] * exp(-p2['kappa'] * (r / beta - p2['s0'])) *
              ksi / p1['gamma2'] * theta)
        y2 = (0.5 * p2['V0'] * exp(-p1['kappa'] * (r / beta - p1['s0'])) /
              ksi / p2['gamma2'] * theta)
        self.energy -= y1 + y2
        f = ((y1 * p2['kappa'] + y2 * p1['kappa']) / beta +
             (y1 + y2) * self.acut * theta * x) * d / r
        self.forces[a1] += f
        self.forces[a2] -= f
        self.sigma1[a1] += (exp(-p2['eta2'] * (r - beta * p2['s0'])) *
                            ksi * theta / p1['gamma1'])
        self.sigma1[a2] += (exp(-p1['eta2'] * (r - beta * p1['s0'])) /
                            ksi * theta / p2['gamma1'])

    def interact2(self, a1, a2, d, r, p1, p2, ksi):
        x = exp(self.acut * (r - self.rc))
        theta = 1.0 / (1.0 + x)
        y1 = (exp(-p2['eta2'] * (r - beta * p2['s0'])) *
              ksi / p1['gamma1'] * theta * self.deds[a1])
        y2 = (exp(-p1['eta2'] * (r - beta * p1['s0'])) /
              ksi / p2['gamma1'] * theta * self.deds[a2])
        f = ((y1 * p2['eta2'] + y2 * p1['eta2']) +
             (y1 + y2) * self.acut * theta * x) * d / r
        self.forces[a1] -= f
        self.forces[a2] += f
