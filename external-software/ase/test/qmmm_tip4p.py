from math import cos, sin

import numpy as np
# import matplotlib.pyplot as plt

import ase.units as units
from ase import Atoms
from ase.calculators.tip4p import TIP4P, epsilon0, sigma0, rOH, angleHOH
from ase.calculators.qmmm import SimpleQMMM, LJInteractions, EIQMMM
from ase.constraints import FixBondLengths
from ase.optimize import BFGS

r = rOH
a = angleHOH * np.pi / 180

# From http://dx.doi.org/10.1063/1.445869
eexp = 6.24 * units.kcal / units.mol
dexp = 2.75
aexp = 46

D = np.linspace(2.5, 3.5, 30)

inter = LJInteractions({('O', 'O'): (epsilon0, sigma0)})

for calc in [TIP4P(),
             SimpleQMMM([0, 1, 2], TIP4P(), TIP4P(), TIP4P()),
             SimpleQMMM([0, 1, 2], TIP4P(), TIP4P(), TIP4P(), vacuum=3.0),
             EIQMMM([0, 1, 2], TIP4P(), TIP4P(), inter),
             EIQMMM([0, 1, 2], TIP4P(), TIP4P(), inter, vacuum=3.0),
             EIQMMM([3, 4, 5], TIP4P(), TIP4P(), inter, vacuum=3.0)]:
    dimer = Atoms('OH2OH2',
                  [(0, 0, 0),
                   (r * cos(a), 0, r * sin(a)),
                   (r, 0, 0),
                   (0, 0, 0),
                   (r * cos(a / 2), r * sin(a / 2), 0),
                   (r * cos(a / 2), -r * sin(a / 2), 0)
                   ])
    dimer.calc = calc
    E = []
    F = []
    for d in D:
        dimer.positions[3:, 0] += d - dimer.positions[3, 0]
        E.append(dimer.get_potential_energy())
        F.append(dimer.get_forces())

    F = np.array(F)

    # plt.plot(D, E)

    F1 = np.polyval(np.polyder(np.polyfit(D, E, 7)), D)
    F2 = F[:, :3, 0].sum(1)
    error = abs(F1 - F2).max()

    dimer.constraints = FixBondLengths([(3 * i + j, 3 * i + (j + 1) % 3)
                                       for i in range(2)
                                        for j in [0, 1, 2]])
    opt = BFGS(dimer,
               trajectory=calc.name + '.traj', logfile=calc.name + 'd.log')
    opt.run(0.001)

    if calc.name == 'tip4p':  # save optimized geom for EIQMMM test
        tip4pdimer = dimer.copy()

    e0 = dimer.get_potential_energy()
    d0 = dimer.get_distance(0, 3)
    R = dimer.positions
    v1 = R[2] - R[3]
    v2 = R[3] - (R[4] + R[5]) / 2
    a0 = np.arccos(np.dot(v1, v2) /
                   (np.dot(v1, v1) * np.dot(v2, v2))**0.5) / np.pi * 180
    fmt = '{0:>25}: {1:.3f} {2:.3f} {3:.3f} {4:.1f}'
    print(fmt.format(calc.name, -min(E), -e0, d0, a0))
    assert abs(e0 + eexp) < 0.002
    assert abs(d0 - dexp) < 0.006
    assert abs(a0 - aexp) < 2.5

# plt.show()

print(fmt.format('reference', 9.999, eexp, dexp, aexp))
