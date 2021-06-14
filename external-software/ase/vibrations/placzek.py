# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np

import ase.units as u
from ase.vibrations.resonant_raman import ResonantRaman

# XXX remove gpaw dependence
from gpaw.lrtddft.spectrum import polarizability


class Placzek(ResonantRaman):
    """Raman spectra within the Placzek approximation."""
    def __init__(self, *args, **kwargs):
        self._approx = 'PlaczekAlpha'
        ResonantRaman.__init__(self, *args, **kwargs)

    def set_approximation(self, value):
        raise ValueError('Approximation can not be set.')

    def read_excitations(self):
        self.ex0E_p = None  # mark as read
        self.exm_r = []
        self.exp_r = []
        for a, i in zip(self.myindices, self.myxyz):
            exname = '%s.%d%s-' % (self.exname, a, i) + self.exext
            self.log('reading ' + exname)
            self.exm_r.append(self.exobj(exname, **self.exkwargs))
            exname = '%s.%d%s+' % (self.exname, a, i) + self.exext
            self.log('reading ' + exname)
            self.exp_r.append(self.exobj(exname, **self.exkwargs))

    def electronic_me_Qcc(self, omega, gamma=0):
        self.read()
        
        self.timer.start('init')
        V_rcc = np.zeros((self.ndof, 3, 3), dtype=complex)
        pre = 1. / (2 * self.delta)
        pre *= u.Hartree * u.Bohr  # e^2Angstrom^2 / eV -> Angstrom^3

        om = omega
        if gamma:
            om += 1j * gamma
        self.timer.stop('init')
        
        self.timer.start('alpha derivatives')
        for i, r in enumerate(self.myr):
            V_rcc[r] = pre * (
                polarizability(self.exp_r[i], om,
                               form=self.dipole_form, tensor=True) -
                polarizability(self.exm_r[i], om,
                               form=self.dipole_form, tensor=True))
        self.timer.stop('alpha derivatives')
 
        # map to modes
        self.comm.sum(V_rcc)
        V_qcc = (V_rcc.T * self.im).T  # units Angstrom^2 / sqrt(amu)
        V_Qcc = np.dot(V_qcc.T, self.modes.T).T
        return V_Qcc


class Profeta(ResonantRaman):
    """Profeta type approximations.

    Reference
    ---------
    Mickael Profeta and Francesco Mauri
    Phys. Rev. B 63 (2000) 245415
    """
    def __init__(self, *args, **kwargs):
        self.set_approximation(kwargs.pop('approximation', 'Profeta'))
        self.nonresonant = kwargs.pop('nonresonant', True)
        ResonantRaman.__init__(self, *args, **kwargs)

    def set_approximation(self, value):
        approx = value.lower()
        if approx in ['profeta', 'placzek', 'p-p']:
            self._approx = value
        else:
            raise ValueError('Please use "Profeta", "Placzek" or "P-P".')
        
    def electronic_me_profeta_rcc(self, omega, gamma=0.1,
                                  energy_derivative=False):
        """Raman spectra in Profeta and Mauri approximation

        Returns
        -------
        Electronic matrix element, unit Angstrom^2
         """
        self.read()

        self.timer.start('amplitudes')

        self.timer.start('init')
        V_rcc = np.zeros((self.ndof, 3, 3), dtype=complex)
        pre = 1. / (2 * self.delta)
        pre *= u.Hartree * u.Bohr  # e^2Angstrom^2 / eV -> Angstrom^3
        self.timer.stop('init')

        def kappa_cc(me_pc, e_p, omega, gamma, form='v'):
            """Kappa tensor after Profeta and Mauri
            PRB 63 (2001) 245415"""
            k_cc = np.zeros((3, 3), dtype=complex)
            for p, me_c in enumerate(me_pc):
                me_cc = np.outer(me_c, me_c.conj())
                k_cc += me_cc / (e_p[p] - omega - 1j * gamma)
                if self.nonresonant:
                    k_cc += me_cc.conj() / (e_p[p] + omega + 1j * gamma)
            return k_cc

        self.timer.start('kappa')
        mr = 0
        for a, i, r in zip(self.myindices, self.myxyz, self.myr):
            if not energy_derivative < 0:
                V_rcc[r] += pre * (
                    kappa_cc(self.expm_rpc[mr], self.ex0E_p,
                             omega, gamma, self.dipole_form) -
                    kappa_cc(self.exmm_rpc[mr], self.ex0E_p,
                             omega, gamma, self.dipole_form))
            if energy_derivative:
                V_rcc[r] += pre * (
                    kappa_cc(self.ex0m_pc, self.expE_rp[mr],
                             omega, gamma, self.dipole_form) -
                    kappa_cc(self.ex0m_pc, self.exmE_rp[mr],
                             omega, gamma, self.dipole_form))
            mr += 1
        self.comm.sum(V_rcc)
        self.timer.stop('kappa')
        self.timer.stop('amplitudes')

        return V_rcc

    def electronic_me_Qcc(self, omega, gamma):
        self.read()
        Vel_rcc = np.zeros((self.ndof, 3, 3), dtype=complex)
        approximation = self.approximation.lower()
        if approximation == 'profeta':
            Vel_rcc += self.electronic_me_profeta_rcc(omega, gamma)
        elif approximation == 'placzek':
            Vel_rcc += self.electronic_me_profeta_rcc(omega, gamma, True)
        elif approximation == 'p-p':
            Vel_rcc += self.electronic_me_profeta_rcc(omega, gamma, -1)
        else:
            raise RuntimeError(
                'Bug: call with {0} should not happen!'.format(
                    self.approximation))

        # map to modes
        self.timer.start('map R2Q')
        V_qcc = (Vel_rcc.T * self.im).T  # units Angstrom^2 / sqrt(amu)
        Vel_Qcc = np.dot(V_qcc.T, self.modes.T).T
        self.timer.stop('map R2Q')

        return Vel_Qcc
