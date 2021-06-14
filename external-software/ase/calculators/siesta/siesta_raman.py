# -*- coding: utf-8 -*-

"""Infrared and Raman intensities using siesta and MBPT_LCAO"""

import pickle
from math import sqrt
from sys import stdout
import numpy as np
import ase.units as units
from ase.parallel import parprint, paropen
from ase.vibrations import Vibrations
from ase.utils import basestring
import warnings

# XXX This class contains much repeated code.  FIXME


class SiestaRaman(Vibrations):
    """
    Class for calculating vibrational modes, infrared and
    non-resonant Raman intensities using finite difference.

    The vibrational modes are calculated from a finite difference
    approximation of the Dynamical matrix and the IR and Ram intensities from
    a finite difference approximation of the gradient of the dipole
    moment. The method is described in:

      D. Porezag, M. R. Pederson:
      "Infrared intensities and Raman-scattering activities within
      density-functional theory",
      Phys. Rev. B 54, 7830 (1996)

    The calculator object (calc) must be Siesta, and the
    pyscf program (nao branch: https://github.com/cfm-mpc/pyscf/tree/nao)
    must be installed.

    >>> calc.get_dipole_moment(atoms)

    In addition to the methods included in the ``Vibrations`` class
    the ``Raman`` as the ``Infrared`` class introduces two methods;
    *get_spectrum()* and *write_spectra()*. The *summary()*, *get_energies()*,
    *get_frequencies()*, *get_spectrum()* and *write_spectra()*
    methods all take an optional *method* keyword.  Use
    method='Frederiksen' to use the method described in:

      T. Frederiksen, M. Paulsson, M. Brandbyge, A. P. Jauho:
      "Inelastic transport theory from first-principles: methodology
      and applications for nanoscale devices",
      Phys. Rev. B 75, 205413 (2007)

    atoms: Atoms object
        The atoms to work on.
    siesta: Siesta calculator
    mbpt_inp: dict
        dictionnary containing the input for the mbpt_lcao program
    indices: list of int
        List of indices of atoms to vibrate.  Default behavior is
        to vibrate all atoms.
    name: str
        Name to use for files.
    delta: float
        Magnitude of displacements.
    nfree: int
        Number of displacements per degree of freedom, 2 or 4 are
        supported. Default is 2 which will displace each atom +delta
        and -delta in each cartesian direction.
    directions: list of int
        Cartesian coordinates to calculate the gradient
        of the dipole moment in.
        For example directions = 2 only dipole moment in the z-direction will
        be considered, whereas for directions = [0, 1] only the dipole
        moment in the xy-plane will be considered. Default behavior is to
        use the dipole moment in all directions.

    freq_pol: float or array of float
        frequency at which the Raman intensity is computed, can be float or array 

    Example:

    See the example test/siesta/mbpt_lcao/script_raman.py
    This example calculate the Raman signal for a CO2 molecule.
    You should get something like,

    ---------------------------------------------------------------------------------------------------------------------------
    Mode    Frequency        Intensity IR        Intensity Raman (real)          Intensity Raman (imag)          Raman Ehanced
    #    meV     cm^-1   (D/Å)^2 amu^-1          A^4 amu^-1                     A^4 amu^-1                      A^4 amu^-1
    ---------------------------------------------------------------------------------------------------------------------------
    0   23.2i    187.1i     0.0005                1.0810                 0.0000                 0.0000
    1   22.7i    183.2i     0.0007                1.1471                 0.0000                 0.0000
    2    4.0i     32.6i     0.0001                0.0054                 0.0000                 0.0000
    3   19.6     158.0      0.0001                1.1790                 0.0000                 0.0000
    4   21.0     169.2      0.0004                1.0894                 0.0000                 0.0000
    5   77.9     628.3      0.4257                0.0060                 0.0000                 0.0000
    6   79.7     642.6      0.4354                0.0022                 0.0000                 0.0000
    7  163.5    1319.0      0.0000               21.7631                 0.0000                 0.0000
    8  294.0    2371.0     12.1479                0.0002                 0.0000                 0.0000
    ---------------------------------------------------------------------------------------------------------------------------

    It can be compared to calculations done with Quantum Espresso (see  test/siesta/mbpt_lcao/raman_espresso)
    that give something like,

    # mode   [cm-1]    [THz]      IR          Raman   depol.fact
    1     -0.01   -0.0002    0.0000         0.4930    0.7500
    2     -0.00   -0.0000    0.0000         0.0018    0.7500
    3      0.00    0.0001    0.0000         0.8202    0.7500
    4      0.00    0.0001    0.0000         0.9076    0.7500
    5      0.01    0.0002    0.0000         1.8576    0.7499
    6      0.07    0.0021    0.0000         0.0001    0.7500
    7    717.64   21.5144    0.5303         0.0000    0.0862
    8   1244.37   37.3052    0.0000        23.8219    0.1038
    9   2206.78   66.1575   12.6139         0.0000    0.6417
    """

    def __init__(self, atoms, siesta, indices=None, name='ram',
                 delta=0.01, nfree=2, directions=None, freq_pol=0.0, **kw):

        Vibrations.__init__(self, atoms, indices=indices, name=name, 
                            delta = delta, nfree=nfree)
        if atoms.constraints:
            warnings.warn('WARNING! \n Your Atoms object is constrained. ' +
                  'Some forces may be unintended set to zero. \n')
        self.name = name + '-d%.3f' % delta
        self.calc = atoms.get_calculator()

        if directions is None:
            self.directions = np.asarray([0, 1, 2])
        else:
            self.directions = np.asarray(directions)
        self.ir = True
        self.ram = True
        self.siesta = siesta

        if isinstance(freq_pol, list):
            self.freq_pol = np.array(freq_pol)
        elif isinstance(freq_pol, float):
            self.freq_pol = np.array([freq_pol])
        elif isinstance(freq_pol, float) or isinstance(freq_pol, np.ndarray):
            self.freq_pol = freq_pol
        else:
            raise ValueError("wrong type for freq_pol, only float, list or array")

        self.pyscf_arg = kw

    def get_polarizability(self):
        if "tddft_iter_tol" in list(self.pyscf_arg.keys()):
            if self.pyscf_arg["tddft_iter_tol"] > 1e-4:
                warnings.warn("tddft_iter_tol > 1e-4, polarizability may not have " +
                              "enough precision. The Raman intensity will not be precise.")
        else:
            self.pyscf_arg["tddft_iter_tol"] = 1e-4

        self.siesta.pyscf_tddft(Edir=np.array([1.0, 1.0, 1.0]), **self.pyscf_arg)
        return self.siesta.results["freq range"], \
               self.siesta.results["polarizability nonin"], \
               self.siesta.results["polarizability inter"]

    def read(self, method='standard', direction='central', inter = True):
        self.method = method.lower()
        self.direction = direction.lower()
        assert self.method in ['standard', 'frederiksen']
        if direction != 'central':
            raise NotImplementedError(
                'Only central difference is implemented at the moment.')

        # Get "static" dipole moment polarizability and forces
        name = '%s.eq.pckl' % self.name
        [forces_zero, dipole_zero, freq_zero,
            noninPol_zero, pol_zero] = pickle.load(open(name, "rb"))
        self.dipole_zero = (sum(dipole_zero**2)**0.5) / units.Debye
        self.force_zero = max([sum((forces_zero[j])**2)**0.5
                               for j in self.indices])
        self.noninPol_zero = noninPol_zero * (units.Bohr)**3  # Ang**3
        self.pol_zero = pol_zero * (units.Bohr)**3  # Ang**3

        ndof = 3 * len(self.indices)
        H = np.empty((ndof, ndof))
        dpdx = np.empty((ndof, 3))
        dadx = np.empty((ndof, self.pol_zero.shape[0], 3, 3), dtype=complex)

        r = 0
        for a in self.indices:
            for i in 'xyz':
                name = '%s.%d%s' % (self.name, a, i)
                [fminus, dminus, frminus, noninpminus, pminus] = pickle.load(
                    open(name + '-.pckl', "rb"))
                [fplus, dplus, frplus, noninpplus, pplus] = pickle.load(
                    open(name + '+.pckl', "rb"))
                if self.nfree == 4:
                    [fminusminus, dminusminus, frminusminus, 
                            noninpminusminus, pminusminus] =\
                                    pickle.load(open(name + '--.pckl', "rb"))
                    [fplusplus, dplusplus, frplusplus, 
                            noninpplusplus, pplusplus] =\
                                    pickle.load(open(name + '++.pckl', "rb"))
                if self.method == 'frederiksen':
                    fminus[a] += -fminus.sum(0)
                    fplus[a] += -fplus.sum(0)
                    if self.nfree == 4:
                        fminusminus[a] += -fminus.sum(0)
                        fplusplus[a] += -fplus.sum(0)
                if self.nfree == 2:
                    H[r] = (fminus - fplus)[self.indices].ravel() / 2.0
                    dpdx[r] = (dminus - dplus)
                    if inter:
                        dadx[r] = (pminus - pplus)
                    else:
                        dadx[r] = (noninpminus - noninpplus)
                if self.nfree == 4:
                    H[r] = (-fminusminus + 8 * fminus - 8 * fplus +
                            fplusplus)[self.indices].ravel() / 12.0
                    dpdx[r] = (-dplusplus + 8 * dplus - 8 * dminus +
                               dminusminus) / 6.0
                    if inter:
                        dadx[r] = (-pplusplus + 8 * pplus - 8 * pminus +
                                   pminusminus) / 6.0
                    else:
                        dadx[r] = (-noninpplusplus + 8 * noninpplus - 8 * noninpminus +
                                   noninpminusminus) / 6.0
                H[r] /= 2 * self.delta
                dpdx[r] /= 2 * self.delta
                dadx[r] /= 2 * self.delta  # polarizability in Ang
                for n in range(3):
                    if n not in self.directions:
                        dpdx[r][n] = 0
                        dpdx[r][n] = 0
                r += 1
        # Calculate eigenfrequencies and eigenvectors
        m = self.atoms.get_masses()
        H += H.copy().T
        self.H = H
        m = self.atoms.get_masses()
        self.im = np.repeat(m[self.indices]**-0.5, 3)
        omega2, modes = np.linalg.eigh(self.im[:, None] * H * self.im)
        self.modes = modes.T.copy()

        # infrared
        # Calculate intensities
        dpdq = np.array([dpdx[j] / sqrt(m[self.indices[j // 3]] *
                                        units._amu / units._me)
                         for j in range(ndof)])
        dpdQ = np.dot(dpdq.T, modes)
        dpdQ = dpdQ.T
        intensities = np.array([sum(dpdQ[j]**2) for j in range(ndof)])
        # Conversion factor:
        s = units._hbar * 1e10 / sqrt(units._e * units._amu)
        self.hnu = s * omega2.astype(complex)**0.5
        # Conversion factor from atomic units to (D/Angstrom)^2/amu.
        conv = (1.0 / units.Debye)**2 * units._amu / units._me
        self.intensities_ir = intensities * conv

        # Raman
        dadq = np.array([(dadx[j, :, :, :] / (units.Bohr**2)) /
                         sqrt(m[self.indices[j // 3]] * units._amu / units._me)
                         for j in range(ndof)])
        dadQ = np.zeros((ndof, 3, 3, dadq.shape[1]), dtype=complex)
        for w in range(dadq.shape[1]):
            dadQ[:, :, :, w] = np.dot(dadq[:, w, :, :].T, modes).T

        ak = (dadQ[:, 0, 0, :] + dadQ[:, 1, 1, :] + dadQ[:, 2, 2, :]) / 3.0
        gk2 = ((dadQ[:, 0, 0, :] - dadQ[:, 1, 1, :])**2 + (dadQ[:, 1, 1, :] -
                dadQ[:, 2, 2, :])**2 + (dadQ[:, 2, 2, :] -
                dadQ[:, 0, 0, :])**2 + 6 * (dadQ[:, 0, 1, :]**2 +
                dadQ[:, 1, 2, :]**2 + dadQ[:, 2, 0, :]**2))

        intensities = np.zeros((ndof, self.freq_pol.size), dtype=np.complex128)
        intensities_ram_enh = np.zeros((ndof, self.freq_pol.size), dtype=np.complex128)

        # calculate the coefficients for calculating Raman Signal
        for j in range(ndof):
            aj = self.get_nearrest_value(self.freq_pol, freq_zero, ak[j, :])
            gj2 = self.get_nearrest_value(self.freq_pol, freq_zero, gk2[j, :])

            intensities[j, :] = (45 * aj**2 + 7 * gj2) / 45.0

        self.intensities_ram = intensities  # Bohr**4 .me**-1
        self.intensities_ram_enh = intensities_ram_enh  # Bohr**4 .me**-1

    def get_nearrest_value(self, val, x_range, y, kind="cubic", prec = 1e-3):
        """
            return the closest value of y at val (interpolating the function)
            val may be a number or an array
        """
        import scipy.interpolate as interp


        func = interp.interp1d(x_range, y, kind = kind)
        new_range = np.arange(x_range[0], x_range[x_range.size-1], prec)
        interpol = func(new_range)

        if isinstance(val, np.ndarray):
            mult = np.zeros(val.shape, dtype=interpol.dtype)

            for i, va in enumerate(val):
                idx = (np.abs(new_range-va)).argmin()
                mult[i] = interpol[idx]
            return mult
        else:
            idx = (np.abs(new_range-val)).argmin()
            return np.array([interpol[idx]])

    def intensity_prefactor(self, intensity_unit):
        if intensity_unit == '(D/A)2/amu':
            return 1.0, '(D/Å)^2 amu^-1'
        elif intensity_unit == 'km/mol':
            # conversion factor from Porezag PRB 54 (1996) 7830
            return 42.255, 'km/mol'
        elif intensity_unit == 'au':
            return 1.0, '   a.u.'
        elif intensity_unit == 'A^4 amu^-1':
            # Quantum espresso units
            return (units.Bohr**4) / (units._me / units._amu), '   A^4 amu^-1'
        else:
            raise RuntimeError('Intensity unit >' + intensity_unit +
                               '< unknown.')

    def summary(self, method='standard', direction='central', freq_pol = 0.0,
                intensity_unit_ir='(D/A)2/amu', intensity_unit_ram='au', log=stdout,
                inter = True):
        hnu = self.get_energies(method, direction, inter=inter)
        s = 0.01 * units._e / units._c / units._hplanck
        iu_ir, iu_string_ir = self.intensity_prefactor(intensity_unit_ir)
        iu_ram, iu_string_ram = self.intensity_prefactor(intensity_unit_ram)
        arr = []
 
        freq_idx = (np.abs(self.freq_pol-freq_pol)).argmin()
        print("index: ", freq_idx)

        if intensity_unit_ir == '(D/A)2/amu':
            iu_format_ir = '%9.4f             '
        elif intensity_unit_ir == 'km/mol':
            iu_string_ir = '   ' + iu_string_ir
            iu_format_ir = ' %7.1f            '
        elif intensity_unit_ir == 'au':
            iu_format_ir = '%.6e              '
        elif intensity_unit_ir == 'A^4 amu^-1':
            iu_format_ir = '%9.4f             '

        if intensity_unit_ram == '(D/A)2/amu':
            iu_format_ram = '%9.4f'
        elif intensity_unit_ram == 'km/mol':
            iu_string_ram = '   ' + iu_string_ram
            iu_format_ram = ' %7.1f'
        elif intensity_unit_ram == 'au':
            iu_format_ram = '%.6e              '
        elif intensity_unit_ram == 'A^4 amu^-1':
            iu_format_ram = '%9.4f              '

        if isinstance(log, basestring):
            log = paropen(log, 'a')

        parprint('---------------------------------------------------------------------------------------------------------------------------', file=log)
        parprint(' Mode    Frequency        Intensity IR        Intensity Raman (real)          Intensity Raman (imag)          Raman Ehanced', file=log)
        parprint('  #    meV     cm^-1   ' + iu_string_ir + '       ' + iu_string_ram +
                 '                  ' + iu_string_ram + '                   ' + iu_string_ram, file=log)
        parprint('---------------------------------------------------------------------------------------------------------------------------', file=log)
        for n, e in enumerate(hnu):
            if e.imag != 0:
                c = 'i'
                e = e.imag
            else:
                c = ' '
                e = e.real
                arr.append([n, 1000 * e, s * e, iu_ir * self.intensities_ir[n],
                            iu_ram * self.intensities_ram[n, freq_idx].real, iu_ram *
                            self.intensities_ram[n, freq_idx].imag])
            parprint(('%3d %6.1f%s  %7.1f%s  ' + iu_format_ir + iu_format_ram +
                      iu_format_ram + iu_format_ram) %
                     (n, 1000 * e, c, s * e, c, iu_ir * self.intensities_ir[n],
                      iu_ram * self.intensities_ram[n, freq_idx].real, iu_ram *
                      self.intensities_ram[n, freq_idx].imag, iu_ram *
                      self.intensities_ram_enh[n, freq_idx].real), file=log)
        parprint(
            '-----------------------------------------------------------------------------------------',
            file=log)
        parprint('Zero-point energy: %.3f eV' % self.get_zero_point_energy(),
                 file=log)
        parprint('Static dipole moment: %.3f D' % self.dipole_zero, file=log)
        parprint('Maximum force on atom in `equilibrium`: %.4f eV/Å' %
                 self.force_zero, file=log)
        parprint(file=log)
        np.savetxt('ram-summary.txt', np.array(arr))

    def write_latex_array(self, fname='vib_latex_array.tex.table', caption='', nb_column=5,
                          hline=False, method='standard', direction='central',
                          intensity_unit_ir='(D/A)2/amu', intensity_unit_ram='au',
                          label='tab_vib', log=stdout, freq_pol=0.0):
        """
        Write the summary into a latex table that can be easily incorporate into a latex file.
        """

        hnu = self.get_energies(method, direction)
        s = 0.01 * units._e / units._c / units._hplanck
        iu_ir, iu_string_ir = self.intensity_prefactor(intensity_unit_ir)
        iu_ram, iu_string_ram = self.intensity_prefactor(intensity_unit_ram)
 
        freq_idx = (np.abs(self.freq_pol-freq_pol)).argmin()


        if intensity_unit_ir == '(D/A)2/amu':
            iu_format_ir = '%9.4f             '
        elif intensity_unit_ir == 'km/mol':
            iu_string_ir = '   ' + iu_string_ir
            iu_format_ir = ' %7.1f            '
        elif intensity_unit_ir == 'au':
            iu_format_ir = '%.6e              '
        elif intensity_unit_ir == 'A^4 amu^-1':
            iu_format_ir = '%9.4f             '

        if intensity_unit_ram == '(D/A)2/amu':
            iu_format_ram = '%9.4f'
        elif intensity_unit_ram == 'km/mol':
            iu_string_ram = '   ' + iu_string_ram
            iu_format_ram = ' %7.1f'
        elif intensity_unit_ram == 'au':
            iu_format_ram = '%.6e              '
        elif intensity_unit_ram == 'A^4 amu^-1':
            iu_format_ram = '%9.4f              '

        if isinstance(log, basestring):
            log = paropen(log, 'a')

        if hline:
            column = "|"
        else:
            column = ""

        for i in range(nb_column + 1):
            if hline:
                column = column + "c|"
            else:
                column = column + "c"

        f = open(fname, 'w')
        f.write("\begin{table}[h] \n")
        f.write("  \caption{" + caption + "} \n")
        f.write("  \begin{center}\n")
        f.write("    \begin{tabular}{" + column + "} \n")

        if hline:
            f.write("     \hline \n")

        f.write('     Mode & Frequency (meV) & Frequency ($cm^{-1}$) & Intensity IR  (' +
                iu_string_ir + ')  & Intensity Raman (' + iu_string_ram + ') \n')
        if hline:
            f.write("   \hline \n")
        for n, e in enumerate(hnu):
            if e.imag != 0:
                c = ' + i'
                e = e.imag
            else:
                c = ' '
                e = e.real
            f.write(('     %3d & %6.1f %s  & %7.1f %s  & ' + iu_format_ir +
                     '  & ' + iu_format_ram + ' \n') % (n, 1000 * e, c, s * e, c,
                                                        iu_ir * self.intensities_ir[n], iu_ram *
                                                        self.intensities_ram[n, freq_idx].real))
            if hline:
                f.write(r"      \hline \n")
        f.write("    \end{tabular} \n")
        f.write("  \end{center} \n")
        f.write(" \label{" + label + "} \n")
        f.write("\end{table}\n")

        f.close()

    def get_spectrum(self, start=800, end=4000, npts=None, width=4,
                     type='Gaussian', method='standard', direction='central',
                     intensity_unit='(D/A)2/amu', normalize=False, freq_pol=0.0):
        """Get raman spectrum.

        The method returns wavenumbers in cm^-1 with corresponding
        absolute infrared intensity.
        Start and end point, and width of the Gaussian/Lorentzian should
        be given in cm^-1.
        normalize=True ensures the integral over the peaks to give the
        intensity.
        """
        name = '%s.eq.pckl' % self.name
        [forces_zero, dipole_zero, freq_zero, noninPol_zero,
            pol_zero] = pickle.load(open(name, "rb"))
 
        freq_idx = (np.abs(self.freq_pol-freq_pol)).argmin()

        frequencies = self.get_frequencies(method, direction).real
        intensities_ir = self.intensities_ir
        intensities_ram = self.intensities_ram[:, freq_idx]
        intensities_ram_enh = self.intensities_ram_enh[:, freq_idx]
        energies, spectrum_ir = self.fold(frequencies, intensities_ir,
                                          start, end, npts, width, type,
                                          normalize)
        energies, spectrum_ram = self.fold(frequencies, intensities_ram.real,
                                           start, end, npts, width, type,
                                           normalize)
        energies, spectrum_ram_enh = self.fold(frequencies, intensities_ram_enh.real,
                                               start, end, npts, width, type,
                                               normalize)
        return energies, spectrum_ir, spectrum_ram, spectrum_ram_enh

    def write_spectra(self, out='ram-spectra.dat', start=800, end=4000,
                      npts=None, width=10, type='Gaussian',
                      method='standard', direction='central',
                      intensity_unit_ir='(D/A)2/amu',
                      intensity_unit_ram='au', normalize=False):
        """Write out raman spectrum to file.
        First column is the wavenumber in cm^-1, the second column the
        absolute infrared intensities, and
        the third column the absorbance scaled so that data runs
        from 1 to 0.  idem for the Rahman spectrum for columns 4 and 5. Start and end
        point, and width of the Gaussian/Lorentzian should be given
        in cm^-1."""
        energies, spectrum_ir, spectrum_ram, spectrum_ram_enh =\
            self.get_spectrum(start, end, npts, width, type, method,
                              direction, normalize)

        # Write out spectrum in file. First column is absolute intensities.
        # Second column is absorbance scaled so that data runs from 1 to 0
        spectrum2_ir = 1. - spectrum_ir / spectrum_ir.max()
        spectrum2_ram = 1. - spectrum_ram / spectrum_ram.max()
        if abs(spectrum_ram_enh.max()) > 0.0:
            spectrum2_ram_enh = 1. - spectrum_ram_enh / spectrum_ram_enh.max()
        else:
            spectrum2_ram_enh = np.zeros(spectrum_ram.shape, dtype = np.float64)

        outdata = np.empty([len(energies), 7])
        outdata.T[0] = energies
        outdata.T[1] = spectrum_ir
        outdata.T[2] = spectrum2_ir
        outdata.T[3] = spectrum_ram
        outdata.T[4] = spectrum2_ram
        outdata.T[5] = spectrum_ram_enh
        outdata.T[6] = spectrum2_ram_enh

        fd = open(out, 'w')
        fd.write('# %s folded, width=%g cm^-1\n' % (type.title(), width))
        iu_ir, iu_string_ir = self.intensity_prefactor(intensity_unit_ir)
        iu_ram, iu_string_ram = self.intensity_prefactor(intensity_unit_ram)

        # if normalize:
        #    iu_string = 'cm ' + iu_string_ir + iu_string_ram
        fd.write('# [cm^-1] %14s\n' %
                 ('[' + iu_string_ir + iu_string_ram + iu_string_ram + ']'))
        for row in outdata:
            fd.write('%.3f  %15.5e  %15.5e %15.5e  %15.5e   %15.5e    %15.5e\n' %
                     (row[0], iu_ir * row[1], row[2], iu_ram * row[3], row[4],
                      iu_ram * row[5], row[6]))
        fd.close()
