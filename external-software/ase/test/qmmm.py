from math import cos, sin, pi

import numpy as np
# import matplotlib.pyplot as plt

import ase.units as units
from ase import Atoms
from ase.calculators.tip3p import TIP3P, epsilon0, sigma0, rOH, angleHOH
from ase.calculators.qmmm import (SimpleQMMM, EIQMMM, LJInteractions,
                                  LJInteractionsGeneral)
from ase.constraints import FixInternals
from ase.optimize import GPMin

r = rOH
a = angleHOH * pi / 180

# From http://dx.doi.org/10.1063/1.445869
eexp = 6.50 * units.kcal / units.mol
dexp = 2.74
aexp = 27

D = np.linspace(2.5, 3.5, 30)

i = LJInteractions({('O', 'O'): (epsilon0, sigma0)})

# General LJ interaction object
sigma_mm = np.array([0, 0, sigma0])
epsilon_mm = np.array([0, 0, epsilon0])
sigma_qm = np.array([0, 0, sigma0])
epsilon_qm = np.array([0, 0, epsilon0])
ig = LJInteractionsGeneral(sigma_qm, epsilon_qm, sigma_mm, epsilon_mm)

for calc in [TIP3P(),
             SimpleQMMM([0, 1, 2], TIP3P(), TIP3P(), TIP3P()),
             SimpleQMMM([0, 1, 2], TIP3P(), TIP3P(), TIP3P(), vacuum=3.0),
             EIQMMM([0, 1, 2], TIP3P(), TIP3P(), i),
             EIQMMM([3, 4, 5], TIP3P(), TIP3P(), i, vacuum=3.0),
             EIQMMM([0, 1, 2], TIP3P(), TIP3P(), i, vacuum=3.0),
             EIQMMM([0, 1, 2], TIP3P(), TIP3P(), ig),
             EIQMMM([3, 4, 5], TIP3P(), TIP3P(), ig, vacuum=3.0),
             EIQMMM([0, 1, 2], TIP3P(), TIP3P(), ig, vacuum=3.0)]:
    dimer = Atoms('H2OH2O',
                  [(r * cos(a), 0, r * sin(a)),
                   (r, 0, 0),
                   (0, 0, 0),
                   (r * cos(a / 2), r * sin(a / 2), 0),
                   (r * cos(a / 2), -r * sin(a / 2), 0),
                   (0, 0, 0)])
    dimer.calc = calc

    E = []
    F = []
    for d in D:
        dimer.positions[3:, 0] += d - dimer.positions[5, 0]
        E.append(dimer.get_potential_energy())
        F.append(dimer.get_forces())

    F = np.array(F)

    # plt.plot(D, E)

    F1 = np.polyval(np.polyder(np.polyfit(D, E, 7)), D)
    F2 = F[:, :3, 0].sum(1)
    error = abs(F1 - F2).max()

    dimer.constraints = FixInternals(
        bonds=[(r, (0, 2)), (r, (1, 2)),
               (r, (3, 5)), (r, (4, 5))],
        angles=[(a, (0, 2, 1)), (a, (3, 5, 4))])
    opt = GPMin(dimer,
               trajectory=calc.name + '.traj', logfile=calc.name + 'd.log')
    opt.run(0.01)

    e0 = dimer.get_potential_energy()
    d0 = dimer.get_distance(2, 5)
    R = dimer.positions
    v1 = R[1] - R[5]
    v2 = R[5] - (R[3] + R[4]) / 2
    a0 = np.arccos(np.dot(v1, v2) /
                   (np.dot(v1, v1) * np.dot(v2, v2))**0.5) / np.pi * 180
    fmt = '{0:>20}: {1:.3f} {2:.3f} {3:.3f} {4:.1f}'
    print(fmt.format(calc.name, -min(E), -e0, d0, a0))
    assert abs(e0 + eexp) < 0.002
    assert abs(d0 - dexp) < 0.01
    assert abs(a0 - aexp) < 4

print(fmt.format('reference', 9.999, eexp, dexp, aexp))

# plt.show()
