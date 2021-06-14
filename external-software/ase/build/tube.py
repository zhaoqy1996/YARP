from __future__ import print_function
from math import sqrt

import numpy as np

from ase.atoms import Atoms
from ase.utils import gcd


def nanotube(n, m, length=1, bond=1.42, symbol='C', verbose=False,
             vacuum=None):
    if n < m:
        m, n = n, m
        sign = -1
    else:
        sign = 1

    nk = 6000
    sq3 = sqrt(3.0)
    a = sq3 * bond
    l2 = n * n + m * m + n * m
    l = sqrt(l2)

    nd = gcd(n, m)
    if (n - m) % (3 * nd) == 0:
        ndr = 3 * nd
    else:
        ndr = nd

    nr = (2 * m + n) // ndr
    ns = -(2 * n + m) // ndr
    nn = 2 * l2 // ndr

    ichk = 0
    if nr == 0:
        n60 = 1
    else:
        n60 = nr * 4

    absn = abs(n60)
    nnp = []
    nnq = []
    for i in range(-absn, absn + 1):
        for j in range(-absn, absn + 1):
            j2 = nr * j - ns * i
            if j2 == 1:
                j1 = m * i - n * j
                if j1 > 0 and j1 < nn:
                    ichk += 1
                    nnp.append(i)
                    nnq.append(j)

    if ichk == 0:
        raise RuntimeError('not found p, q strange!!')
    if ichk >= 2:
        raise RuntimeError('more than 1 pair p, q strange!!')

    nnnp = nnp[0]
    nnnq = nnq[0]

    if verbose:
        print('the symmetry vector is', nnnp, nnnq)

    lp = nnnp * nnnp + nnnq * nnnq + nnnp * nnnq
    r = a * sqrt(lp)
    c = a * l
    t = sq3 * c / ndr

    if 2 * nn > nk:
        raise RuntimeError('parameter nk is too small!')

    rs = c / (2.0 * np.pi)

    if verbose:
        print('radius=', rs, t)

    q1 = np.arctan((sq3 * m) / (2 * n + m))
    q2 = np.arctan((sq3 * nnnq) / (2 * nnnp + nnnq))
    q3 = q1 - q2

    q4 = 2.0 * np.pi / nn
    q5 = bond * np.cos((np.pi / 6.0) - q1) / c * 2.0 * np.pi

    h1 = abs(t) / abs(np.sin(q3))
    h2 = bond * np.sin((np.pi / 6.0) - q1)

    ii = 0
    x, y, z = [], [], []
    for i in range(nn):
        x1, y1, z1 = 0, 0, 0

        k = np.floor(i * abs(r) / h1)
        x1 = rs * np.cos(i * q4)
        y1 = rs * np.sin(i * q4)
        z1 = (i * abs(r) - k * h1) * np.sin(q3)
        kk2 = abs(np.floor((z1 + 0.0001) / t))
        if z1 >= t - 0.0001:
            z1 -= t * kk2
        elif z1 < 0:
            z1 += t * kk2
        ii += 1

        x.append(x1)
        y.append(y1)
        z.append(z1)
        z3 = (i * abs(r) - k * h1) * np.sin(q3) - h2
        ii += 1

        if z3 >= 0 and z3 < t:
            x2 = rs * np.cos(i * q4 + q5)
            y2 = rs * np.sin(i * q4 + q5)
            z2 = (i * abs(r) - k * h1) * np.sin(q3) - h2
            x.append(x2)
            y.append(y2)
            z.append(z2)
        else:
            x2 = rs * np.cos(i * q4 + q5)
            y2 = rs * np.sin(i * q4 + q5)
            z2 = (i * abs(r) - (k + 1) * h1) * np.sin(q3) - h2
            kk = abs(np.floor(z2 / t))
            if z2 >= t - 0.0001:
                z2 -= t * kk
            elif z2 < 0:
                z2 += t * kk
            x.append(x2)
            y.append(y2)
            z.append(z2)

    ntotal = 2 * nn
    X = []
    for i in range(ntotal):
        X.append([x[i], y[i], sign * z[i]])

    if length > 1:
        xx = X[:]
        for mnp in range(2, length + 1):
            for i in range(len(xx)):
                X.append(xx[i][:2] + [xx[i][2] + (mnp - 1) * t])

    transvec = t
    numatom = ntotal * length
    diameter = rs * 2
    chiralangle = np.arctan((sq3 * n) / (2 * m + n)) / np.pi * 180

    cell = [[0, 0, 0], [0, 0, 0], [0, 0, length * t]]
    atoms = Atoms(symbol + str(numatom),
                  positions=X,
                  cell=cell,
                  pbc=[False, False, True])
    if vacuum:
        atoms.center(vacuum, axis=(0, 1))
    if verbose:
        print('translation vector =', transvec)
        print('diameter = ', diameter)
        print('chiral angle = ', chiralangle)
    return atoms
