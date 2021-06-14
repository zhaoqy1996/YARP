from __future__ import print_function
import functools
import warnings

from ase.utils import convert_string_to_fd

import numpy as np


def get_band_gap(calc, direct=False, spin=None, output='-'):
    warnings.warn('Please use ase.dft.bandgap.bandgap() instead!')
    gap, (s1, k1, n1), (s2, k2, n2) = bandgap(calc, direct, spin, output)
    ns = calc.get_number_of_spins()
    if ns == 2 and spin is None:
        return gap, (s1, k1), (s2, k2)
    return gap, k1, k2


def bandgap(calc=None, direct=False, spin=None, output='-',
            eigenvalues=None, efermi=None, kpts=None):
    """Calculates the band-gap.

    Parameters:

    calc: Calculator object
        Electronic structure calculator object.
    direct: bool
        Calculate direct band-gap.
    spin: int or None
        For spin-polarized systems, you can use spin=0 or spin=1 to look only
        at a single spin-channel.
    output: file descriptor
        Use output=None for no text output or '-' for stdout (default).
    eigenvalues: ndarray of shape (nspin, nkpt, nband) or (nkpt, nband)
        Eigenvalues.
    efermi: float
        Fermi level (defaults to 0.0).
    kpts: ndarray of shape (nkpt, 3)
        For pretty text output only.

    Returns a (gap, p1, p2) tuple where p1 and p2 are tuples of indices of the
    valence and conduction points (s, k, n).

    Example:

    >>> gap, p1, p2 = bandgap(silicon.calc)
    Gap: 1.2 eV
    Transition (v -> c):
        [0.000, 0.000, 0.000] -> [0.500, 0.500, 0.000]
    >>> print(gap, p1, p2)
    1.2 (0, 0, 3), (0, 5, 4)
    >>> gap, p1, p2 = bandgap(silicon.calc, direct=True)
    Direct gap: 3.4 eV
    Transition at: [0.000, 0.000, 0.000]
    >>> print(gap, p1, p2)
    3.4 (0, 0, 3), (0, 0, 4)
    """

    if calc:
        kpts = calc.get_ibz_k_points()
        nk = len(kpts)
        ns = calc.get_number_of_spins()
        eigenvalues = np.array([[calc.get_eigenvalues(kpt=k, spin=s)
                                 for k in range(nk)]
                                for s in range(ns)])
        if efermi is None:
            efermi = calc.get_fermi_level()

    efermi = efermi or 0.0

    e_skn = eigenvalues - efermi
    if eigenvalues.ndim != 3:
        e_skn = e_skn[np.newaxis]

    ns, nk, nb = e_skn.shape

    N_sk = (e_skn < 0.0).sum(2)  # number of occupied bands
    e_skn = np.array([[e_skn[s, k, N_sk[s, k] - 1:N_sk[s, k] + 1]
                       for k in range(nk)]
                      for s in range(ns)])
    ev_sk = e_skn[:, :, 0]  # valence band
    ec_sk = e_skn[:, :, 1]  # conduction band

    s1 = s2 = k1 = k2 = n1 = n2 = None

    if ns == 1:
        s1 = 0
        s2 = 0
        if N_sk[0].ptp() > 0:
            # Some band must be crossing the fermi-level
            gap = 0.0
        else:
            gap, k1, k2 = find_gap(ev_sk[0], ec_sk[0], direct)
            n1 = N_sk[0, 0] - 1
            n2 = n1 + 1
    elif spin is None:
        if (N_sk.ptp(axis=1) > 0).any():
            # Some band must be crossing the fermi-level
            gap = 0.0
        else:
            gap, k1, k2 = find_gap(ev_sk.ravel(), ec_sk.ravel(), direct)
            if direct:
                # Check also spin flips:
                for s in [0, 1]:
                    g, k, _ = find_gap(ev_sk[s], ec_sk[1 - s], direct)
                    if g < gap:
                        gap = g
                        k1 = k + nk * s
                        k2 = k + nk * (1 - s)

            if gap > 0.0:
                s1, k1 = divmod(k1, nk)
                s2, k2 = divmod(k2, nk)
                n1 = N_sk[s1, k1] - 1
                n2 = N_sk[s2, k2]
            else:
                s1 = None
                s2 = None

    else:
        if N_sk[spin].ptp() > 0:
            # Some band must be crossing the fermi-level
            gap = 0.0
        else:
            gap, k1, k2 = find_gap(ev_sk[spin], ec_sk[spin], direct)
            s1 = spin
            s2 = spin
            n1 = N_sk[s1, k1] - 1
            n2 = n1 + 1

    if output is not None:
        def skn(s, k, n):
            """Convert k or (s, k) to string."""
            if kpts is None:
                return '(s={}, k={}, n={})'.format(s, k, n)
            return '(s={}, k={}, n={}, [{:.3f}, {:.3f}, {:.3f}])'.format(
                s, k, n, *kpts[k])

        p = functools.partial(print, file=convert_string_to_fd(output))
        if spin is not None:
            p('spin={}: '.format(spin), end='')
        if gap == 0.0:
            p('No gap!')
        elif direct:
            p('Direct gap: {:.3f} eV'.format(gap))
            if s1 == s2:
                p('Transition at:', skn(s1, k1, n1))
            else:
                p('Transition at:', skn('{}->{}'.format(s1, s2), k1, n1))
        else:
            p('Gap: {:.3f} eV'.format(gap))
            p('Transition (v -> c):')
            p(' ', skn(s1, k1, n1), '->', skn(s2, k2, n2))

    if eigenvalues.ndim != 3:
        p1 = (k1, n1)
        p2 = (k2, n2)
    else:
        p1 = (s1, k1, n1)
        p2 = (s2, k2, n2)

    return gap, p1, p2


def find_gap(ev_k, ec_k, direct):
    """Helper function."""
    if direct:
        gap_k = ec_k - ev_k
        k = gap_k.argmin()
        return gap_k[k], k, k
    kv = ev_k.argmax()
    kc = ec_k.argmin()
    return ec_k[kc] - ev_k[kv], kv, kc
