from __future__ import print_function
"""Modules for calculating thermochemical information from computational
outputs."""

import os
import sys
import numpy as np

from ase import units


class ThermoChem:
    """Base class containing common methods used in thermochemistry
    calculations."""

    def get_ZPE_correction(self):
        """Returns the zero-point vibrational energy correction in eV."""
        zpe = 0.
        for energy in self.vib_energies:
            zpe += 0.5 * energy
        return zpe

    def _vibrational_energy_contribution(self, temperature):
        """Calculates the change in internal energy due to vibrations from
        0K to the specified temperature for a set of vibrations given in
        eV and a temperature given in Kelvin. Returns the energy change
        in eV."""
        kT = units.kB * temperature
        dU = 0.
        for energy in self.vib_energies:
            dU += energy / (np.exp(energy / kT) - 1.)
        return dU

    def _vibrational_entropy_contribution(self, temperature):
        """Calculates the entropy due to vibrations for a set of vibrations
        given in eV and a temperature given in Kelvin.  Returns the entropy
        in eV/K."""
        kT = units.kB * temperature
        S_v = 0.
        for energy in self.vib_energies:
            x = energy / kT
            S_v += x / (np.exp(x) - 1.) - np.log(1. - np.exp(-x))
        S_v *= units.kB
        return S_v

    def _vprint(self, text):
        """Print output if verbose flag True."""
        if self.verbose:
            sys.stdout.write(text + os.linesep)


class HarmonicThermo(ThermoChem):
    """Class for calculating thermodynamic properties in the approximation
    that all degrees of freedom are treated harmonically. Often used for
    adsorbates.

    Inputs:

    vib_energies : list
        a list of the harmonic energies of the adsorbate (e.g., from
        ase.vibrations.Vibrations.get_energies). The number of
        energies should match the number of degrees of freedom of the
        adsorbate; i.e., 3*n, where n is the number of atoms. Note that
        this class does not check that the user has supplied the correct
        number of energies. Units of energies are eV.
    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this
        class can be interpreted as the energy corrections)
    """

    def __init__(self, vib_energies, potentialenergy=0.):
        self.vib_energies = vib_energies
        # Check for imaginary frequencies.
        if sum(np.iscomplex(self.vib_energies)):
            raise ValueError('Imaginary vibrational energies are present.')
        else:
            self.vib_energies = np.real(self.vib_energies)  # clear +0.j

        self.potentialenergy = potentialenergy

    def get_internal_energy(self, temperature, verbose=True):
        """Returns the internal energy, in eV, in the harmonic approximation
        at a specified temperature (K)."""

        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.3f eV'
        write('Internal energy components at T = %.2f K:' % temperature)
        write('=' * 31)

        U = 0.

        write(fmt % ('E_pot', self.potentialenergy))
        U += self.potentialenergy

        zpe = self.get_ZPE_correction()
        write(fmt % ('E_ZPE', zpe))
        U += zpe

        dU_v = self._vibrational_energy_contribution(temperature)
        write(fmt % ('Cv_harm (0->T)', dU_v))
        U += dU_v

        write('-' * 31)
        write(fmt % ('U', U))
        write('=' * 31)
        return U

    def get_entropy(self, temperature, verbose=True):
        """Returns the entropy, in eV/K, in the harmonic approximation
        at a specified temperature (K)."""

        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.7f eV/K%13.3f eV'
        write('Entropy components at T = %.2f K:' % temperature)
        write('=' * 49)
        write('%15s%13s     %13s' % ('', 'S', 'T*S'))

        S = 0.

        S_v = self._vibrational_entropy_contribution(temperature)
        write(fmt % ('S_harm', S_v, S_v * temperature))
        S += S_v

        write('-' * 49)
        write(fmt % ('S', S, S * temperature))
        write('=' * 49)
        return S

    def get_helmholtz_energy(self, temperature, verbose=True):
        """Returns the Helmholtz free energy, in eV, in the harmonic
        approximation at a specified temperature (K)."""

        self.verbose = True
        write = self._vprint

        U = self.get_internal_energy(temperature, verbose=verbose)
        write('')
        S = self.get_entropy(temperature, verbose=verbose)
        F = U - temperature * S

        write('')
        write('Free energy components at T = %.2f K:' % temperature)
        write('=' * 23)
        fmt = '%5s%15.3f eV'
        write(fmt % ('U', U))
        write(fmt % ('-T*S', -temperature * S))
        write('-' * 23)
        write(fmt % ('F', F))
        write('=' * 23)
        return F


class HinderedThermo(ThermoChem):
    """Class for calculating thermodynamic properties in the hindered
    translator and hindered rotor model where all but three degrees of
    freedom are treated as harmonic vibrations, two are treated as
    hindered translations, and one is treated as a hindered rotation.

    Inputs:

    vib_energies : list
        a list of all the vibrational energies of the adsorbate (e.g., from
        ase.vibrations.Vibrations.get_energies). The number of energies
        should match the number of degrees of freedom of the adsorbate;
        i.e., 3*n, where n is the number of atoms. Note that this class does
        not check that the user has supplied the correct number of energies.
        Units of energies are eV.
    trans_barrier_energy : float
        the translational energy barrier in eV. This is the barrier for an
        adsorbate to diffuse on the surface.
    rot_barrier_energy : float
        the rotational energy barrier in eV. This is the barrier for an
        adsorbate to rotate about an axis perpendicular to the surface.
    sitedensity : float
        density of surface sites in cm^-2
    rotationalminima : integer
        the number of equivalent minima for an adsorbate's full rotation.
        For example, 6 for an adsorbate on an fcc(111) top site
    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this class
        can be interpreted as the energy corrections)
    mass : float
        the mass of the adsorbate in amu (if mass is unspecified, then it will
        be calculated from the atoms class)
    inertia : float
        the reduced moment of inertia of the adsorbate in amu*Ang^-2
        (if inertia is unspecified, then it will be calculated from the
        atoms class)
    atoms : an ASE atoms object
        used to calculate rotational moments of inertia and molecular mass
    symmetrynumber : integer
        symmetry number of the adsorbate. This is the number of symmetric arms
        of the adsorbate and depends upon how it is bound to the surface.
        For example, propane bound through its end carbon has a symmetry
        number of 1 but propane bound through its middle carbon has a symmetry
        number of 2. (if symmetrynumber is unspecified, then the default is 1)
    """

    def __init__(self, vib_energies, trans_barrier_energy, rot_barrier_energy,
                 sitedensity, rotationalminima, potentialenergy=0.,
                 mass=None, inertia=None, atoms=None, symmetrynumber=1):
        self.vib_energies = sorted(vib_energies, reverse=True)[:-3]
        self.trans_barrier_energy = trans_barrier_energy * units._e
        self.rot_barrier_energy = rot_barrier_energy * units._e
        self.area = 1. / sitedensity / 100.0**2
        self.rotationalminima = rotationalminima
        self.potentialenergy = potentialenergy
        self.atoms = atoms
        self.symmetry = symmetrynumber

        if (mass or atoms) and (inertia or atoms):
            if mass:
                self.mass = mass * units._amu
            elif atoms:
                self.mass = np.sum(atoms.get_masses()) * units._amu
            if inertia:
                self.inertia = inertia * units._amu / units.m**2
            elif atoms:
                self.inertia = (atoms.get_moments_of_inertia()[2] *
                                units._amu / units.m**2)
        else:
            raise RuntimeError('Either mass and inertia of the '
                               'adsorbate must be specified or '
                               'atoms must be specified.')

        # Make sure no imaginary frequencies remain.
        if sum(np.iscomplex(self.vib_energies)):
            raise ValueError('Imaginary frequencies are present.')
        else:
            self.vib_energies = np.real(self.vib_energies)  # clear +0.j

        # Calculate hindered translational and rotational frequencies
        self.freq_t = np.sqrt(self.trans_barrier_energy / (2 * self.mass *
                                                           self.area))
        self.freq_r = 1. / (2 * np.pi) * np.sqrt(self.rotationalminima**2 *
                                                 self.rot_barrier_energy /
                                                 (2 * self.inertia))

    def get_internal_energy(self, temperature, verbose=True):
        """Returns the internal energy (including the zero point energy),
        in eV, in the hindered translator and hindered rotor model at a
        specified temperature (K)."""

        from scipy.special import iv

        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.3f eV'
        write('Internal energy components at T = %.2f K:' % temperature)
        write('=' * 31)

        U = 0.

        write(fmt % ('E_pot', self.potentialenergy))
        U += self.potentialenergy

        # Translational Energy
        T_t = units._k * temperature / (units._hplanck * self.freq_t)
        R_t = self.trans_barrier_energy / (units._hplanck * self.freq_t)
        dU_t = 2 * (-1. / 2 - 1. / T_t / (2 + 16 * R_t) + R_t / 2 / T_t -
                    R_t / 2 / T_t *
                    iv(1, R_t / 2 / T_t) / iv(0, R_t / 2 / T_t) +
                    1. / T_t / (np.exp(1. / T_t) - 1))
        dU_t *= units.kB * temperature
        write(fmt % ('E_trans', dU_t))
        U += dU_t

        # Rotational Energy
        T_r = units._k * temperature / (units._hplanck * self.freq_r)
        R_r = self.rot_barrier_energy / (units._hplanck * self.freq_r)
        dU_r = (-1. / 2 - 1. / T_r / (2 + 16 * R_r) + R_r / 2 / T_r -
                R_r / 2 / T_r *
                iv(1, R_r / 2 / T_r) / iv(0, R_r / 2 / T_r) +
                1. / T_r / (np.exp(1. / T_r) - 1))
        dU_r *= units.kB * temperature
        write(fmt % ('E_rot', dU_r))
        U += dU_r

        # Vibrational Energy
        dU_v = self._vibrational_energy_contribution(temperature)
        write(fmt % ('E_vib', dU_v))
        U += dU_v

        # Zero Point Energy
        dU_zpe = self.get_zero_point_energy()
        write(fmt % ('E_ZPE', dU_zpe))
        U += dU_zpe

        write('-' * 31)
        write(fmt % ('U', U))
        write('=' * 31)
        return U

    def get_zero_point_energy(self, verbose=True):
        """Returns the zero point energy, in eV, in the hindered
        translator and hindered rotor model"""

        zpe_t = 2 * (1. / 2 * self.freq_t * units._hplanck / units._e)
        zpe_r = 1. / 2 * self.freq_r * units._hplanck / units._e
        zpe_v = self.get_ZPE_correction()
        zpe = zpe_t + zpe_r + zpe_v
        return zpe

    def get_entropy(self, temperature, verbose=True):
        """Returns the entropy, in eV/K, in the hindered translator
        and hindered rotor model at a specified temperature (K)."""

        from scipy.special import iv

        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.7f eV/K%13.3f eV'
        write('Entropy components at T = %.2f K:' % temperature)
        write('=' * 49)
        write('%15s%13s     %13s' % ('', 'S', 'T*S'))

        S = 0.

        # Translational Entropy
        T_t = units._k * temperature / (units._hplanck * self.freq_t)
        R_t = self.trans_barrier_energy / (units._hplanck * self.freq_t)
        S_t = 2 * (-1. / 2 + 1. / 2 * np.log(np.pi * R_t / T_t) -
                   R_t / 2 / T_t *
                   iv(1, R_t / 2 / T_t) / iv(0, R_t / 2 / T_t) +
                   np.log(iv(0, R_t / 2 / T_t)) +
                   1. / T_t / (np.exp(1. / T_t) - 1) -
                   np.log(1 - np.exp(-1. / T_t)))
        S_t *= units.kB
        write(fmt % ('S_trans', S_t, S_t * temperature))
        S += S_t

        # Rotational Entropy
        T_r = units._k * temperature / (units._hplanck * self.freq_r)
        R_r = self.rot_barrier_energy / (units._hplanck * self.freq_r)
        S_r = (-1. / 2 + 1. / 2 * np.log(np.pi * R_r / T_r) -
               np.log(self.symmetry) -
               R_r / 2 / T_r * iv(1, R_r / 2 / T_r) / iv(0, R_r / 2 / T_r) +
               np.log(iv(0, R_r / 2 / T_r)) +
               1. / T_r / (np.exp(1. / T_r) - 1) -
               np.log(1 - np.exp(-1. / T_r)))
        S_r *= units.kB
        write(fmt % ('S_rot', S_r, S_r * temperature))
        S += S_r

        # Vibrational Entropy
        S_v = self._vibrational_entropy_contribution(temperature)
        write(fmt % ('S_vib', S_v, S_v * temperature))
        S += S_v

        # Concentration Related Entropy
        N_over_A = np.exp(1. / 3) * (10.0**5 /
                                     (units._k * temperature))**(2. / 3)
        S_c = 1 - np.log(N_over_A) - np.log(self.area)
        S_c *= units.kB
        write(fmt % ('S_con', S_c, S_c * temperature))
        S += S_c

        write('-' * 49)
        write(fmt % ('S', S, S * temperature))
        write('=' * 49)
        return S

    def get_helmholtz_energy(self, temperature, verbose=True):
        """Returns the Helmholtz free energy, in eV, in the hindered
        translator and hindered rotor model at a specified temperature
        (K)."""

        self.verbose = True
        write = self._vprint

        U = self.get_internal_energy(temperature, verbose=verbose)
        write('')
        S = self.get_entropy(temperature, verbose=verbose)
        F = U - temperature * S

        write('')
        write('Free energy components at T = %.2f K:' % temperature)
        write('=' * 23)
        fmt = '%5s%15.3f eV'
        write(fmt % ('U', U))
        write(fmt % ('-T*S', -temperature * S))
        write('-' * 23)
        write(fmt % ('F', F))
        write('=' * 23)
        return F


class IdealGasThermo(ThermoChem):
    """Class for calculating thermodynamic properties of a molecule
    based on statistical mechanical treatments in the ideal gas
    approximation.

    Inputs for enthalpy calculations:

    vib_energies : list
        a list of the vibrational energies of the molecule (e.g., from
        ase.vibrations.Vibrations.get_energies). The number of vibrations
        used is automatically calculated by the geometry and the number of
        atoms. If more are specified than are needed, then the lowest
        numbered vibrations are neglected. If either atoms or natoms is
        unspecified, then uses the entire list. Units are eV.
    geometry : 'monatomic', 'linear', or 'nonlinear'
        geometry of the molecule
    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this
        class can be interpreted as the energy corrections)
    natoms : integer
        the number of atoms, used along with 'geometry' to determine how
        many vibrations to use. (Not needed if an atoms object is supplied
        in 'atoms' or if the user desires the entire list of vibrations
        to be used.)

    Extra inputs needed for entropy / free energy calculations:

    atoms : an ASE atoms object
        used to calculate rotational moments of inertia and molecular mass
    symmetrynumber : integer
        symmetry number of the molecule. See, for example, Table 10.1 and
        Appendix B of C. Cramer "Essentials of Computational Chemistry",
        2nd Ed.
    spin : float
        the total electronic spin. (0 for molecules in which all electrons
        are paired, 0.5 for a free radical with a single unpaired electron,
        1.0 for a triplet with two unpaired electrons, such as O_2.)
    """

    def __init__(self, vib_energies, geometry, potentialenergy=0.,
                 atoms=None, symmetrynumber=None, spin=None, natoms=None):
        self.potentialenergy = potentialenergy
        self.geometry = geometry
        self.atoms = atoms
        self.sigma = symmetrynumber
        self.spin = spin
        if natoms is None:
            if atoms:
                natoms = len(atoms)
        # Cut the vibrations to those needed from the geometry.
        if natoms:
            if geometry == 'nonlinear':
                self.vib_energies = vib_energies[-(3 * natoms - 6):]
            elif geometry == 'linear':
                self.vib_energies = vib_energies[-(3 * natoms - 5):]
            elif geometry == 'monatomic':
                self.vib_energies = []
        else:
            self.vib_energies = vib_energies
        # Make sure no imaginary frequencies remain.
        if sum(np.iscomplex(self.vib_energies)):
            raise ValueError('Imaginary frequencies are present.')
        else:
            self.vib_energies = np.real(self.vib_energies)  # clear +0.j
        self.referencepressure = 1.0e5  # Pa

    def get_enthalpy(self, temperature, verbose=True):
        """Returns the enthalpy, in eV, in the ideal gas approximation
        at a specified temperature (K)."""

        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.3f eV'
        write('Enthalpy components at T = %.2f K:' % temperature)
        write('=' * 31)

        H = 0.

        write(fmt % ('E_pot', self.potentialenergy))
        H += self.potentialenergy

        zpe = self.get_ZPE_correction()
        write(fmt % ('E_ZPE', zpe))
        H += zpe

        Cv_t = 3. / 2. * units.kB  # translational heat capacity (3-d gas)
        write(fmt % ('Cv_trans (0->T)', Cv_t * temperature))
        H += Cv_t * temperature

        if self.geometry == 'nonlinear':  # rotational heat capacity
            Cv_r = 3. / 2. * units.kB
        elif self.geometry == 'linear':
            Cv_r = units.kB
        elif self.geometry == 'monatomic':
            Cv_r = 0.
        write(fmt % ('Cv_rot (0->T)', Cv_r * temperature))
        H += Cv_r * temperature

        dH_v = self._vibrational_energy_contribution(temperature)
        write(fmt % ('Cv_vib (0->T)', dH_v))
        H += dH_v

        Cp_corr = units.kB * temperature
        write(fmt % ('(C_v -> C_p)', Cp_corr))
        H += Cp_corr

        write('-' * 31)
        write(fmt % ('H', H))
        write('=' * 31)
        return H

    def get_entropy(self, temperature, pressure, verbose=True):
        """Returns the entropy, in eV/K, in the ideal gas approximation
        at a specified temperature (K) and pressure (Pa)."""

        if self.atoms is None or self.sigma is None or self.spin is None:
            raise RuntimeError('atoms, symmetrynumber, and spin must be '
                               'specified for entropy and free energy '
                               'calculations.')
        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.7f eV/K%13.3f eV'
        write('Entropy components at T = %.2f K and P = %.1f Pa:' %
              (temperature, pressure))
        write('=' * 49)
        write('%15s%13s     %13s' % ('', 'S', 'T*S'))

        S = 0.0

        # Translational entropy (term inside the log is in SI units).
        mass = sum(self.atoms.get_masses()) * units._amu  # kg/molecule
        S_t = (2 * np.pi * mass * units._k *
               temperature / units._hplanck**2)**(3.0 / 2)
        S_t *= units._k * temperature / self.referencepressure
        S_t = units.kB * (np.log(S_t) + 5.0 / 2.0)
        write(fmt % ('S_trans (1 bar)', S_t, S_t * temperature))
        S += S_t

        # Rotational entropy (term inside the log is in SI units).
        if self.geometry == 'monatomic':
            S_r = 0.0
        elif self.geometry == 'nonlinear':
            inertias = (self.atoms.get_moments_of_inertia() * units._amu /
                        (10.0**10)**2)  # kg m^2
            S_r = np.sqrt(np.pi * np.product(inertias)) / self.sigma
            S_r *= (8.0 * np.pi**2 * units._k * temperature /
                    units._hplanck**2)**(3.0 / 2.0)
            S_r = units.kB * (np.log(S_r) + 3.0 / 2.0)
        elif self.geometry == 'linear':
            inertias = (self.atoms.get_moments_of_inertia() * units._amu /
                        (10.0**10)**2)  # kg m^2
            inertia = max(inertias)  # should be two identical and one zero
            S_r = (8 * np.pi**2 * inertia * units._k * temperature /
                   self.sigma / units._hplanck**2)
            S_r = units.kB * (np.log(S_r) + 1.)
        write(fmt % ('S_rot', S_r, S_r * temperature))
        S += S_r

        # Electronic entropy.
        S_e = units.kB * np.log(2 * self.spin + 1)
        write(fmt % ('S_elec', S_e, S_e * temperature))
        S += S_e

        # Vibrational entropy.
        S_v = self._vibrational_entropy_contribution(temperature)
        write(fmt % ('S_vib', S_v, S_v * temperature))
        S += S_v

        # Pressure correction to translational entropy.
        S_p = - units.kB * np.log(pressure / self.referencepressure)
        write(fmt % ('S (1 bar -> P)', S_p, S_p * temperature))
        S += S_p

        write('-' * 49)
        write(fmt % ('S', S, S * temperature))
        write('=' * 49)
        return S

    def get_gibbs_energy(self, temperature, pressure, verbose=True):
        """Returns the Gibbs free energy, in eV, in the ideal gas
        approximation at a specified temperature (K) and pressure (Pa)."""

        self.verbose = verbose
        write = self._vprint

        H = self.get_enthalpy(temperature, verbose=verbose)
        write('')
        S = self.get_entropy(temperature, pressure, verbose=verbose)
        G = H - temperature * S

        write('')
        write('Free energy components at T = %.2f K and P = %.1f Pa:' %
              (temperature, pressure))
        write('=' * 23)
        fmt = '%5s%15.3f eV'
        write(fmt % ('H', H))
        write(fmt % ('-T*S', -temperature * S))
        write('-' * 23)
        write(fmt % ('G', G))
        write('=' * 23)
        return G


class CrystalThermo(ThermoChem):
    """Class for calculating thermodynamic properties of a crystalline
    solid in the approximation that a lattice of N atoms behaves as a
    system of 3N independent harmonic oscillators.

    Inputs:

    phonon_DOS : list
        a list of the phonon density of states,
        where each value represents the phonon DOS at the vibrational energy
        value of the corresponding index in phonon_energies.

    phonon_energies : list
        a list of the range of vibrational energies (hbar*omega) over which
        the phonon density of states has been evaluated. This list should be
        the same length as phonon_DOS and integrating phonon_DOS over
        phonon_energies should yield approximately 3N, where N is the number
        of atoms per unit cell. If the first element of this list is
        zero-valued it will be deleted along with the first element of
        phonon_DOS. Units of vibrational energies are eV.

    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this
        class can be interpreted as the energy corrections)

    formula_units : int
        the number of formula units per unit cell. If unspecified, the
        thermodynamic quantities calculated will be listed on a
        per-unit-cell basis.
    """

    def __init__(self, phonon_DOS, phonon_energies,
                 formula_units=None, potentialenergy=0.):
        self.phonon_energies = phonon_energies
        self.phonon_DOS = phonon_DOS

        if formula_units:
            self.formula_units = formula_units
            self.potentialenergy = potentialenergy / formula_units
        else:
            self.formula_units = 0
            self.potentialenergy = potentialenergy

    def get_internal_energy(self, temperature, verbose=True):
        """Returns the internal energy, in eV, of crystalline solid
        at a specified temperature (K)."""

        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.4f eV'
        if self.formula_units == 0:
            write('Internal energy components at '
                  'T = %.2f K,\non a per-unit-cell basis:' % temperature)
        else:
            write('Internal energy components at '
                  'T = %.2f K,\non a per-formula-unit basis:' % temperature)
        write('=' * 31)

        U = 0.

        omega_e = self.phonon_energies
        dos_e = self.phonon_DOS
        if omega_e[0] == 0.:
            omega_e = np.delete(omega_e, 0)
            dos_e = np.delete(dos_e, 0)

        write(fmt % ('E_pot', self.potentialenergy))
        U += self.potentialenergy

        zpe_list = omega_e / 2.
        if self.formula_units == 0:
            zpe = np.trapz(zpe_list * dos_e, omega_e)
        else:
            zpe = np.trapz(zpe_list * dos_e, omega_e) / self.formula_units
        write(fmt % ('E_ZPE', zpe))
        U += zpe

        B = 1. / (units.kB * temperature)
        E_vib = omega_e / (np.exp(omega_e * B) - 1.)
        if self.formula_units == 0:
            E_phonon = np.trapz(E_vib * dos_e, omega_e)
        else:
            E_phonon = np.trapz(E_vib * dos_e, omega_e) / self.formula_units
        write(fmt % ('E_phonon', E_phonon))
        U += E_phonon

        write('-' * 31)
        write(fmt % ('U', U))
        write('=' * 31)
        return U

    def get_entropy(self, temperature, verbose=True):
        """Returns the entropy, in eV/K, of crystalline solid
        at a specified temperature (K)."""

        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.7f eV/K%13.4f eV'
        if self.formula_units == 0:
            write('Entropy components at '
                  'T = %.2f K,\non a per-unit-cell basis:' % temperature)
        else:
            write('Entropy components at '
                  'T = %.2f K,\non a per-formula-unit basis:' % temperature)
        write('=' * 49)
        write('%15s%13s     %13s' % ('', 'S', 'T*S'))

        omega_e = self.phonon_energies
        dos_e = self.phonon_DOS
        if omega_e[0] == 0.:
            omega_e = np.delete(omega_e, 0)
            dos_e = np.delete(dos_e, 0)

        B = 1. / (units.kB * temperature)
        S_vib = (omega_e / (temperature * (np.exp(omega_e * B) - 1.)) -
                 units.kB * np.log(1. - np.exp(-omega_e * B)))
        if self.formula_units == 0:
            S = np.trapz(S_vib * dos_e, omega_e)
        else:
            S = np.trapz(S_vib * dos_e, omega_e) / self.formula_units

        write('-' * 49)
        write(fmt % ('S', S, S * temperature))
        write('=' * 49)
        return S

    def get_helmholtz_energy(self, temperature, verbose=True):
        """Returns the Helmholtz free energy, in eV, of crystalline solid
        at a specified temperature (K)."""

        self.verbose = True
        write = self._vprint

        U = self.get_internal_energy(temperature, verbose=verbose)
        write('')
        S = self.get_entropy(temperature, verbose=verbose)
        F = U - temperature * S

        write('')
        if self.formula_units == 0:
            write('Helmholtz free energy components at '
                  'T = %.2f K,\non a per-unit-cell basis:' % temperature)
        else:
            write('Helmholtz free energy components at '
                  'T = %.2f K,\non a per-formula-unit basis:' % temperature)
        write('=' * 23)
        fmt = '%5s%15.4f eV'
        write(fmt % ('U', U))
        write(fmt % ('-T*S', -temperature * S))
        write('-' * 23)
        write(fmt % ('F', F))
        write('=' * 23)
        return F
