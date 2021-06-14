# -*- coding: utf-8 -*-

from __future__ import print_function, division
import sys
import numpy as np
from itertools import combinations_with_replacement

import ase.units as u
from ase.parallel import parprint, paropen
from ase.vibrations import Vibrations
from ase.vibrations.resonant_raman import ResonantRaman
from ase.vibrations.franck_condon import FranckCondonOverlap
from ase.vibrations.franck_condon import FranckCondonRecursive


class Albrecht(ResonantRaman):
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        all from ResonantRaman.__init__
        combinations: int
            Combinations to consider for multiple excitations.
            Default is 1, possible 2
        skip: int
            Number of first transitions to exclude. Default 0,
            recommended: 5 for linear molecules, 6 for other molecules
        nm: int
            Number of intermediate m levels to consider, default 20
        """
        self.combinations = kwargs.pop('combinations', 1)
        self.skip = kwargs.pop('skip', 0)
        self.nm = kwargs.pop('nm', 20)
        approximation = kwargs.pop('approximation', 'Albrecht')

        ResonantRaman.__init__(self, *args, **kwargs)

        self.set_approximation(approximation)

    def set_approximation(self, value):
        approx = value.lower()
        if approx in ['albrecht', 'albrecht b', 'albrecht c', 'albrecht bc']:
            if not self.overlap:
                raise ValueError('Overlaps are needed')
        elif not approx == 'albrecht a':
            raise ValueError('Please use "Albrecht" or "Albrecht A/B/C/BC"')
        self._approx = value

    def read(self, method='standard', direction='central'):
        ResonantRaman.read(self, method, direction)

        # single transitions and their occupation
        om_Q = self.om_Q[self.skip:]
        om_v = om_Q
        ndof = len(om_Q)
        n_vQ = np.eye(ndof, dtype=int)
        
        l_Q = range(ndof)
        ind_v = list(combinations_with_replacement(l_Q, 1))
        
        if self.combinations > 1:
            if not self.combinations == 2:
                raise NotImplementedError

            for c in range(2, self.combinations + 1):
                ind_v += list(combinations_with_replacement(l_Q, c))

            nv = len(ind_v)
            n_vQ = np.zeros((nv, ndof), dtype=int)
            om_v = np.zeros((nv), dtype=float)
            for j, wt in enumerate(ind_v):
                for i in wt:
                    n_vQ[j, i] += 1
            om_v = n_vQ.dot(om_Q)

        self.ind_v = ind_v
        self.om_v = om_v
        self.n_vQ = n_vQ  # how many of each
        self.d_vQ = np.where(n_vQ > 0, 1, 0)  # do we have them ?

    def get_energies(self, method='standard', direction='central'):
        Vibrations.get_energies(self, method, direction)
        return self.om_v

    def _collect_r(self, arr_ro, oshape, dtype):
        """Collect an array that is distributed."""
        if len(self.myr) == self.ndof: # serial
            return arr_ro
        data_ro = np.zeros([self.ndof] + oshape, dtype)
        if len(arr_ro):
            data_ro[self.slize] = arr_ro
        self.comm.sum(data_ro)
        return data_ro
        
    def Huang_Rhys_factors(self, forces_r):
        """Evaluate Huang-Rhys factors derived from forces."""
        self.timer.start('Huang-Rhys')
        assert(len(forces_r.flat) == self.ndof)

        # solve the matrix equation for the equilibrium displacements
        X_q = np.linalg.solve(self.im[:, None] * self.H * self.im,
                              forces_r.flat * self.im)
        d_Q = np.dot(self.modes, X_q)

        # Huang-Rhys factors S
        s = 1.e-20 / u.kg / u.C / u._hbar**2
        self.timer.stop('Huang-Rhys')
        return s * d_Q**2 * self.om_Q / 2.

    def displacements(self, forces_r):
        """Evaluate unitless displacements from forces"""
        self.timer.start('displacements')
        assert(len(forces_r.flat) == self.ndof)

        # solve the matrix equation for the equilibrium displacements
        X_q = np.linalg.solve(self.im[:, None] * self.H * self.im,
                              forces_r.flat * self.im)
        d_Q = np.dot(self.modes, X_q)  # unit eV / sqrt(amu) / Angstrom
        self.timer.stop('displacements')

        s = 1.e-20 / u.kg / u.C / u._hbar**2
        return d_Q * np.sqrt(s * self.om_Q)

    def omegaLS(self, omega, gamma):
        omL = omega + 1j * gamma
        omS_Q = omL - self.om_Q
        return omL, omS_Q

    def init_parallel_excitations(self):
        """Init for paralellization over excitations."""
        n_p = len(self.ex0E_p)

        # collect excited state forces
        exF_pr = self._collect_r(self.exF_rp, [n_p], self.ex0E_p.dtype).T

        # select your work load
        myn = -(-n_p // self.comm.size)  # ceil divide
        rank = self.comm.rank
        s = slice(myn * rank, myn * (rank + 1))
        return n_p, range(n_p)[s], exF_pr
    
    def meA(self, omega, gamma=0.1):
        """Evaluate Albrecht A term.

        Returns
        -------
        Full Albrecht A matrix element. Unit: e^2 Angstrom^2 / eV
        """
        self.read()

        self.timer.start('AlbrechtA')

        if not hasattr(self, 'fcr'):
            self.fcr = FranckCondonRecursive()

        omL = omega + 1j * gamma
        omS_Q = omL - self.om_Q
        
        n_p, myp, exF_pr = self.init_parallel_excitations()

        m_Qcc = np.zeros((self.ndof, 3, 3), dtype=complex)
        for p in myp:
            energy = self.ex0E_p[p]
            d_Q = self.displacements(exF_pr[p])
            energy_Q = energy - self.om_Q * d_Q**2 / 2.
            me_cc = np.outer(self.ex0m_pc[p], self.ex0m_pc[p].conj())

            wm_Q = np.zeros((self.ndof), dtype=complex)
            wp_Q = np.zeros((self.ndof), dtype=complex)
            for m in range(self.nm):
                self.timer.start('0mm1')
                fco_Q = self.fcr.direct0mm1(m, d_Q)
                self.timer.stop('0mm1')
                
                self.timer.start('weight_Q')
                e_Q = energy_Q + m * self.om_Q
                wm_Q += fco_Q / (e_Q - omL)
                wp_Q += fco_Q / (e_Q + omS_Q)
                self.timer.stop('weight_Q')
            self.timer.start('einsum')
            m_Qcc += np.einsum('a,bc->abc', wm_Q, me_cc)
            m_Qcc += np.einsum('a,bc->abc', wp_Q, me_cc.conj())
            self.timer.stop('einsum')
        self.comm.sum(m_Qcc)
                
        self.timer.stop('AlbrechtA')
        return m_Qcc  # e^2 Angstrom^2 / eV

    def meAmult(self, omega, gamma=0.1):
        """Evaluate Albrecht A term.

        Returns
        -------
        Full Albrecht A matrix element. Unit: e^2 Angstrom^2 / eV
        """
        self.read()

        self.timer.start('AlbrechtA')

        if not hasattr(self, 'fcr'):
            self.fcr = FranckCondonRecursive()

        omL = omega + 1j * gamma
        omS_v = omL - self.om_v
        nv = len(self.om_v)
        om_Q = self.om_Q[self.skip:]
        nQ = len(om_Q)

        # n_v:
        #     how many FC factors are involved
        # nvib_ov:
        #     delta functions to switch contributions depending on order o
        # ind_ov:
        #     Q indicees
        # n_ov:
        #     # of vibrational excitations
        n_v = self.d_vQ.sum(axis=1)  # multiplicity
        
        nvib_ov = np.empty((self.combinations, nv), dtype=int)
        om_ov = np.zeros((self.combinations, nv), dtype=float)
        n_ov = np.zeros((self.combinations, nv), dtype=int)
        d_ovQ = np.zeros((self.combinations, nv, nQ), dtype=int)
        for o in range(self.combinations):
            nvib_ov[o] = np.array(n_v == (o + 1))
            for v in range(nv):
                try:
                    om_ov[o, v] = om_Q[self.ind_v[v][o]]
                    d_ovQ[o, v, self.ind_v[v][o]] = 1
                except IndexError:
                    pass
        # XXXX change ????
        n_ov[0] = self.n_vQ.max(axis=1)
        n_ov[1] = nvib_ov[1]
        
        n_p, myp, exF_pr = self.init_parallel_excitations()

        m_vcc = np.zeros((nv, 3, 3), dtype=complex)
        for p in myp:
            energy = self.ex0E_p[p]
            d_Q = self.displacements(exF_pr[p])[self.skip:]
            S_Q = d_Q**2 / 2.
            energy_v = energy - self.d_vQ.dot(om_Q * S_Q)
            me_cc = np.outer(self.ex0m_pc[p], self.ex0m_pc[p].conj())

            # Franck-Condon factors
            self.timer.start('0mm1/2')
            fco1_mQ = np.empty((self.nm, nQ), dtype=float)
            fco2_mQ = np.empty((self.nm, nQ), dtype=float)
            for m in range(self.nm):
                fco1_mQ[m] = self.fcr.direct0mm1(m, d_Q)
                fco2_mQ[m] = self.fcr.direct0mm2(m, d_Q)
            self.timer.stop('0mm1/2')

            wm_v = np.zeros((nv), dtype=complex)
            wp_v = np.zeros((nv), dtype=complex)
            for m in range(self.nm):
                self.timer.start('0mm1/2')
                fco1_v = np.where(n_ov[0] == 2,
                                  d_ovQ[0].dot(fco2_mQ[m]),
                                  d_ovQ[0].dot(fco1_mQ[m]))
                self.timer.stop('0mm1/2')

                self.timer.start('weight_Q')
                em_v = energy_v + m * om_ov[0]
                # multiples of same kind
                fco_v = nvib_ov[0] * fco1_v
                wm_v += fco_v / (em_v - omL)
                wp_v += fco_v / (em_v + omS_v)
                if nvib_ov[1].any():
                    # multiples of mixed type
                    for n in range(self.nm):
                        fco2_v = d_ovQ[1].dot(fco1_mQ[n])
                        e_v = em_v + n * om_ov[1]
                        ## print('e_v', e_v[:3])
                        fco_v = nvib_ov[1] * fco1_v * fco2_v
                        wm_v += fco_v / (e_v - omL)
                        wp_v += fco_v / (e_v + omS_v)
                self.timer.stop('weight_Q')
            self.timer.start('einsum')
            m_vcc += np.einsum('a,bc->abc', wm_v, me_cc)
            m_vcc += np.einsum('a,bc->abc', wp_v, me_cc.conj())
            self.timer.stop('einsum')
        self.comm.sum(m_vcc)
                
        self.timer.stop('AlbrechtA')
        return m_vcc  # e^2 Angstrom^2 / eV

    def meBC(self, omega, gamma=0.1,
             term='BC'):
        """Evaluate Albrecht BC term.

        Returns
        -------
        Full Albrecht BC matrix element.
        Unit: e^2 Angstrom / eV / sqrt(amu)
        """
        self.read()

        self.timer.start('AlbrechtBC')
        self.timer.start('initialize')
        if not hasattr(self, 'fco'):
            self.fco = FranckCondonOverlap()

        omL = omega + 1j * gamma
        omS_Q = omL - self.om_Q

        # excited state forces
        n_p, myp, exF_pr = self.init_parallel_excitations()
        # derivatives after normal coordinates
        exdmdr_rpc = self._collect_r(
            self.exdmdr_rpc, [n_p, 3], self.ex0m_pc.dtype)
        dmdq_qpc = (exdmdr_rpc.T * self.im).T  # unit e / sqrt(amu)
        dmdQ_Qpc = np.dot(dmdq_qpc.T, self.modes.T).T  # unit e / sqrt(amu)
        self.timer.stop('initialize')

        me_Qcc = np.zeros((self.ndof, 3, 3), dtype=complex)
        for p in myp:
            energy = self.ex0E_p[p]
            S_Q = self.Huang_Rhys_factors(exF_pr[p])
            # relaxed excited state energy
            ## n_vQ = np.where(self.n_vQ > 0, 1, 0)
            ## energy_v = energy - n_vQ.dot(self.om_Q * S_Q)
            energy_Q = energy - self.om_Q * S_Q

            ## me_cc = np.outer(self.ex0m_pc[p], self.ex0m_pc[p].conj())
            m_c = self.ex0m_pc[p]  # e Angstrom
            dmdQ_Qc = dmdQ_Qpc[:, p]  # e / sqrt(amu)

            wBLS_Q = np.zeros((self.ndof), dtype=complex)
            wBSL_Q = np.zeros((self.ndof), dtype=complex)
            wCLS_Q = np.zeros((self.ndof), dtype=complex)
            wCSL_Q = np.zeros((self.ndof), dtype=complex)
            for m in range(self.nm):
                self.timer.start('0mm1/2')
                f0mmQ1_Q = (self.fco.directT0(m, S_Q) +
                            np.sqrt(2) * self.fco.direct0mm2(m, S_Q))
                f0Qmm1_Q = self.fco.direct(1, m, S_Q)
##                if (self.n_vQ > 1).any():
##                    fco2_Q = self.fco.direct0mm2(m, S_Q)
                self.timer.stop('0mm1/2')
                
                self.timer.start('weight_Q')
                em_Q = energy_Q + m * self.om_Q
                wBLS_Q += f0mmQ1_Q / (em_Q - omL)
                wBSL_Q += f0Qmm1_Q / (em_Q - omL)
                wCLS_Q += f0mmQ1_Q / (em_Q + omS_Q)
                wCSL_Q += f0Qmm1_Q / (em_Q + omS_Q)
                self.timer.stop('weight_Q')
            self.timer.start('einsum')
            # unit e^2 Angstrom / sqrt(amu)
            mdmdQ_Qcc = np.einsum('a,bc->bac', m_c, dmdQ_Qc.conj())
            dmdQm_Qcc = np.einsum('ab,c->abc', dmdQ_Qc, m_c.conj())
            if 'B' in term:
                me_Qcc += np.multiply(wBLS_Q, mdmdQ_Qcc.T).T
                me_Qcc += np.multiply(wBSL_Q, dmdQm_Qcc.T).T
            if 'C' in term:
                me_Qcc += np.multiply(wCLS_Q, mdmdQ_Qcc.T).T
                me_Qcc += np.multiply(wCSL_Q, dmdQm_Qcc.T).T
            self.timer.stop('einsum')
        self.comm.sum(me_Qcc)
                
        self.timer.stop('AlbrechtBC')
        return me_Qcc  # unit e^2 Angstrom / eV / sqrt(amu)

    def electronic_me_Qcc(self, omega, gamma):
        """Evaluate an electronic matric element."""
        self.read()
        approx = self.approximation.lower()
        assert(self.combinations == 1)
        Vel_Qcc = np.zeros((len(self.om_Q), 3, 3), dtype=complex)
        if approx == 'albrecht a' or approx == 'albrecht':
            Vel_Qcc += self.meA(omega, gamma)  # e^2 Angstrom^2 / eV
            # divide through pre-factor
            with np.errstate(divide='ignore'):
                Vel_Qcc *= np.where(self.vib01_Q > 0,
                                    1. / self.vib01_Q, 0)[:, None, None]
            # -> e^2 Angstrom / eV / sqrt(amu)
        if approx == 'albrecht bc' or approx == 'albrecht':
            Vel_Qcc += self.meBC(omega, gamma)  # e^2 Angstrom / eV / sqrt(amu)
        if approx == 'albrecht b':
            Vel_Qcc += self.meBC(omega, gamma, term='B')
        if approx == 'albrecht c':
            Vel_Qcc = self.meBC(omega, gamma, term='C')

        Vel_Qcc *= u.Hartree * u.Bohr  # e^2 Angstrom^2 / eV -> Angstrom^3

        return Vel_Qcc  # Angstrom^2 / sqrt(amu)

    def me_Qcc(self, omega, gamma):
        """Full matrix element"""
        self.read()
        approx = self.approximation.lower()
        nv = len(self.om_v)
        V_vcc = np.zeros((nv, 3, 3), dtype=complex)
        if approx == 'albrecht a' or approx == 'albrecht':
            if self.combinations == 1:
                # e^2 Angstrom^2 / eV
                V_vcc += self.meA(omega, gamma)[self.skip:]
            else:
                V_vcc += self.meAmult(omega, gamma)
        if approx == 'albrecht bc' or approx == 'albrecht':
            if self.combinations == 1:
                vel_vcc = self.meBC(omega, gamma)
                V_vcc += vel_vcc * self.vib01_Q[:, None, None]
            else:
                vel_vcc = self.meBCmult(omega, gamma)
                V_vcc = 0 
        elif approx == 'albrecht b':
            assert(self.combinations == 1)
            vel_vcc = self.meBC(omega, gamma, term='B')
            V_vcc = vel_vcc * self.vib01_Q[:, None, None]
        if approx == 'albrecht c':
            assert(self.combinations == 1)
            vel_vcc = self.meBC(omega, gamma, term='C')
            V_vcc = vel_vcc * self.vib01_Q[:, None, None]

        return V_vcc  # e^2 Angstrom^2 / eV
    
    def summary(self, omega=0, gamma=0,
                method='standard', direction='central',
                log=sys.stdout):
        """Print summary for given omega [eV]"""
        if self.combinations > 1:
            return self.extended_summary()
        
        om_v = self.get_energies(method, direction)
        intensities = self.absolute_intensity(omega, gamma)[self.skip:]

        if isinstance(log, str):
            log = paropen(log, 'a')

        parprint('-------------------------------------', file=log)
        parprint(' excitation at ' + str(omega) + ' eV', file=log)
        parprint(' gamma ' + str(gamma) + ' eV', file=log)
        parprint(' approximation:', self.approximation, file=log)
        parprint(' Mode    Frequency        Intensity', file=log)
        parprint('  #    meV     cm^-1      [A^4/amu]', file=log)
        parprint('-------------------------------------', file=log)
        for n, e in enumerate(om_v):
            if e.imag != 0:
                c = 'i'
                e = e.imag
            else:
                c = ' '
                e = e.real
            parprint('%3d %6.1f   %7.1f%s  %9.1f' %
                     (n, 1000 * e, e / u.invcm, c, intensities[n]),
                     file=log)
        parprint('-------------------------------------', file=log)
        parprint('Zero-point energy: %.3f eV' % self.get_zero_point_energy(),
                 file=log)

    def extended_summary(self, omega=0, gamma=0,
                         method='standard', direction='central',
                         log=sys.stdout):
        """Print summary for given omega [eV]"""
        om_v = self.get_energies(method, direction)
        intens_v = self.intensity(omega, gamma)
        
        if isinstance(log, str):
            log = paropen(log, 'a')

        parprint('-------------------------------------', file=log)
        parprint(' excitation at ' + str(omega) + ' eV', file=log)
        parprint(' gamma ' + str(gamma) + ' eV', file=log)
        parprint(' approximation:', self.approximation, file=log)
        parprint(' observation:', self.observation, file=log)
        parprint(' Mode    Frequency        Intensity', file=log)
        parprint('  #    meV     cm^-1      [e^4A^4/eV^2]', file=log)
        parprint('-------------------------------------', file=log)
        for v, e in enumerate(om_v):
            parprint(self.ind_v[v], '{0:6.1f}   {1:7.1f} {2:9.1f}'.format(
                1000 * e, e / u.invcm, 1e9 * intens_v[v]),
                file=log)
        parprint('-------------------------------------', file=log)
        parprint('Zero-point energy: %.3f eV' % self.get_zero_point_energy(),
                 file=log)
