"""Definition of the XrDebye class.

This module defines the XrDebye class for calculation
of X-ray scattering properties from atomic cluster
using Debye formula.
Also contains routine for calculation of atomic form factors and
X-ray wavelength dict.
"""

from __future__ import print_function
from math import exp, pi, sin, sqrt, cos, acos
import numpy as np


from ase.data import atomic_numbers

# Table (1) of
# D. WAASMAIER AND A. KIRFEL, Acta Cryst. (1995). A51, 416-431
waasmaier = {
     #       a1         b1         a2         b2         a3         b3          a4         b4         a5         b5         c
     'C': [ 2.657506, 14.780758,  1.078079,  0.776775,  1.490909, 42.086843, -4.241070,  -0.000294,  0.713791, 0.239535,   4.297983],
     'N': [11.893780,  0.000158,  3.277479, 10.232723,  1.858092, 30.344690,  0.858927,   0.656065,  0.912985, 0.217287, -11.804902],
     'O': [ 2.960427, 14.182259,  2.5088111, 5.936858,  0.637053,  0.112726,  0.722838,  34.958481,  1.142756, 0.390240,   0.027014],
     'P': [ 1.950541,  0.908139,  4.146930, 27.044953,  1.494560,  0.071280,  1.522042,  67.520190,  5.729711, 1.981173,   0.155233],
     'S': [ 6.372157,  1.514347,  5.154568, 22.092528,  1.473732,  0.061373,  1.635073,  55.445176,  1.209372, 0.646925,   0.154722],
    'Cl': [ 1.446071,  0.052357,  6.870609,  1.193165,  6.151801, 18.343416,  1.750347,  46.398394,  0.634168, 0.401005,   0.146773],
    'Ni': [13.521865,  4.077277,  6.947285,  0.286763,  3.866028, 14.622634,  2.135900,  71.966078,  4.284731, 0.004437,  -2.762697],
    'Cu': [14.014192,  3.738280,  4.784577,  0.003744,  5.056806, 13.034982,  1.457971,  72.554793,  6.932996, 0.265666,  -3.774477],
    'Pd': [ 6.121511,  0.062549,  4.784063,  0.784031, 16.631683,  8.751391,  4.318258,  34.489983, 13.246773, 0.784031,   0.883099],
    'Ag': [ 6.073874,  0.055333, 17.155437,  7.896512,  4.173344, 28.443739,  0.852238, 110.376108, 17.988685, 0.716809,   0.756603],
    'Pt': [31.273891,  1.316992, 18.445441,  8.797154, 17.063745,  0.124741,  5.555933,  40.177994,  1.575270, 1.316997,   4.050394],
    'Au': [16.777389,  0.122737, 19.317156,  8.621570, 32.979682,  1.256902,  5.595453,  38.008821, 10.576854, 0.000601,  -6.279078],
}

wavelengths = {
    'CuKa1': 1.5405981,
    'CuKa2': 1.54443,
    'CuKb1': 1.39225,
    'WLa1': 1.47642,
    'WLa2': 1.48748
}


class XrDebye(object):
    """
    Class for calculation of XRD or SAXS patterns.
    """
    def __init__(self, atoms, wavelength, damping=0.04,
                 method='Iwasa', alpha=1.01, warn=True):
        """
        Initilize the calculation of X-ray diffraction patterns

        Parameters:

        atoms: ase.Atoms
            atoms object for which calculation will be performed.

        wavelength: float, Angstrom
            X-ray wavelength in Angstrom. Used for XRD and to setup dumpings.

        damping : float, Angstrom**2
            thermal damping factor parameter (B-factor).

        method: {'Iwasa'}
            method of calculation (damping and atomic factors affected).

            If set to 'Iwasa' than angular damping and q-dependence of
            atomic factors are used.

            For any other string there will be only thermal damping
            and constant atomic factors (`f_a(q) = Z_a`).

        alpha: float
            parameter for angular damping of scattering intensity.
            Close to 1.0 for unplorized beam.

        warn: boolean
            flag to show warning if atomic factor can't be calculated
        """
        self.wavelength = wavelength
        self.damping = damping
        self.mode = ''
        self.method = method
        self.alpha = alpha
        self.warn = warn

        self.twotheta_list = []
        self.q_list = []
        self.intensity_list = []

        self.atoms = atoms
        # TODO: setup atomic form factors if method != 'Iwasa'

    def set_damping(self, damping):
        """ set B-factor for thermal damping """
        self.damping = damping

    def get(self, s):
        r"""Get the powder x-ray (XRD) scattering intensity
        using the Debye-Formula at single point.

        Parameters:

        s: float, in inverse Angstrom
            scattering vector value (`s = q / 2\pi`).

        Returns:
            Intensity at given scattering vector `s`.
        """

        pre = exp(-self.damping * s**2 / 2)

        if self.method == 'Iwasa':
            sinth = self.wavelength * s / 2.
            positive = 1. - sinth**2
            if positive < 0:
                positive = 0
            costh = sqrt(positive)
            cos2th = cos(2. * acos(costh))
            pre *= costh / (1. + self.alpha * cos2th**2)

        f = {}
        def atomic(symbol):
            """
            get atomic factor, using cache.
            """
            if symbol not in f:
                if self.method == 'Iwasa':
                    f[symbol] = self.get_waasmaier(symbol, s)
                else:
                    f[symbol] = atomic_numbers[symbol]
            return f[symbol]

        I = 0.
        fa = []  # atomic factors list
        for a in self.atoms:
            fa.append(atomic(a.symbol))

        pos = self.atoms.get_positions()  # positions of atoms
        fa = np.array(fa)  # atomic factors array

        for i in range(len(self.atoms)):
            vr = pos - pos[i]
            I += np.sum(fa[i] * fa * np.sinc(2 * s * np.sqrt(np.sum(vr * vr, axis=1))))

        return pre * I

    def get_waasmaier(self, symbol, s):
        r"""Scattering factor for free atoms.

        Parameters:

        symbol: string
            atom element symbol.

        s: float, in inverse Angstrom
            scattering vector value (`s = q / 2\pi`).

        Returns:
            Intensity at given scattering vector `s`.

        Note:
            for hydrogen will be returned zero value."""
        if symbol == 'H':
            # XXXX implement analytical H
            return 0
        elif symbol in waasmaier:
            abc = waasmaier[symbol]
            f = abc[10]
            s2 = s * s
            for i in range(5):
                f += abc[2 * i] * exp(-abc[2 * i + 1] * s2)
            return f
        if self.warn:
            print('<xrdebye::get_atomic> Element', symbol, 'not available')
        return 0

    def calc_pattern(self, x=None, mode='XRD'):
        r"""
        Calculate X-ray diffraction pattern or
        small angle X-ray scattering pattern.

        Parameters:

        x: float array
            points where intensity will be calculated.
            XRD - 2theta values, in degrees;
            SAXS - q values in 1/A
            (`q = 2 \pi \cdot s = 4 \pi \sin( \theta) / \lambda`).
            If ``x`` is ``None`` then default values will be used.

        mode: {'XRD', 'SAXS'}
            the mode of calculation: X-ray diffraction (XRD) or
            small-angle scattering (SAXS).

        Returns:
            list of intensities calculated for values given in ``x``.
        """
        self.mode = mode.upper()
        assert(mode in ['XRD', 'SAXS'])

        result = []
        if mode == 'XRD':
            if x is None:
                self.twotheta_list = np.linspace(15, 55, 100)
            else:
                self.twotheta_list = x
            self.q_list = []
            print('#2theta\tIntensity')
            for twotheta in self.twotheta_list:
                s = 2 * sin(twotheta * pi / 180 / 2.0) / self.wavelength
                result.append(self.get(s))
                print('%.3f\t%f' % (twotheta, result[-1]))
        elif mode == 'SAXS':
            if x is None:
                self.twotheta_list = np.logspace(-3, -0.3, 100)
            else:
                self.q_list = x
            self.twotheta_list = []
            print('#q\tIntensity')
            for q in self.q_list:
                s = q / (2 * pi)
                result.append(self.get(s))
                print('%.4f\t%f' % (q, result[-1]))
        self.intensity_list = np.array(result)
        return self.intensity_list

    def write_pattern(self, filename):
        """ Save calculated data to file specified by ``filename`` string."""
        f = open(filename, 'w')
        f.write('# Wavelength = %f\n' % self.wavelength)
        if self.mode == 'XRD':
            x, y = self.twotheta_list, self.intensity_list
            f.write('# 2theta \t Intesity\n')
        elif self.mode == 'SAXS':
            x, y = self.q_list, self.intensity_list
            f = open(filename, 'w')
            f.write('# q(1/A)\tIntesity\n')
        else:
            f.close()
            raise Exception('No data available, call calc_pattern() first.')

        for i in range(len(x)):
            f.write('  %f\t%f\n' % (x[i], y[i]))

        f.close()

    def plot_pattern(self, filename=None, show=None, ax=None):
        """ Plot XRD or SAXS depending on filled data

        Uses Matplotlib to plot pattern. Use *show=True* to
        show the figure and *filename='abc.png'* or
        *filename='abc.eps'* to save the figure to a file.

        Returns:
            ``matplotlib.axes.Axes`` object."""

        import matplotlib.pyplot as plt

        if filename is None and show is None:
            show = True

        if ax is None:
            plt.clf()  # clear figure
            ax = plt.gca()

        if self.mode == 'XRD':
            x, y = np.array(self.twotheta_list), np.array(self.intensity_list)
            ax.plot(x, y / np.max(y), '.-')
            ax.set_xlabel('2$\\theta$')
            ax.set_ylabel('Intensity')
        elif self.mode == 'SAXS':
            x, y = np.array(self.q_list), np.array(self.intensity_list)
            ax.loglog(x, y / np.max(y), '.-')
            ax.set_xlabel('q, 1/Angstr.')
            ax.set_ylabel('Intensity')
        else:
            raise Exception('No data available, call calc_pattern() first')

        if show:
            plt.show()
        if filename is not None:
            fig = ax.get_figure()
            fig.savefig(filename)

        return ax
