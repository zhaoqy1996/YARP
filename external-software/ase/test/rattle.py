import ase.units as units
from ase.calculators.tip3p import TIP3P, epsilon0, sigma0, rOH, angleHOH
from ase.calculators.qmmm import SimpleQMMM, EIQMMM, LJInteractions
from ase.data.s22 import create_s22_system as s22
from ase.md.verlet import VelocityVerlet
from ase.constraints import FixBondLengths

i = LJInteractions({('O', 'O'): (epsilon0, sigma0)})

for calc in [TIP3P(),
             SimpleQMMM([0, 1, 2], TIP3P(), TIP3P(), TIP3P()),
             EIQMMM([0, 1, 2], TIP3P(), TIP3P(), i)]:
    dimer = s22('Water_dimer')

    for m in [0, 3]:
        dimer.set_angle(m + 1, m, m + 2, angleHOH)
        dimer.set_distance(m, m + 1, rOH, fix=0)
        dimer.set_distance(m, m + 2, rOH, fix=0)

    fixOH1 = [(3 * i, 3 * i + 1) for i in range(2)]
    fixOH2 = [(3 * i, 3 * i + 2) for i in range(2)]
    fixHH = [(3 * i + 1, 3 * i + 2) for i in range(2)]
    dimer.set_constraint(FixBondLengths(fixOH1+fixOH2+fixHH))

    dimer.calc = calc

    e = dimer.get_potential_energy()
    md = VelocityVerlet(dimer, 2.0 * units.fs,
                        trajectory=calc.name + '.traj',
                        logfile=calc.name + '.log',
                        loginterval=20)
    md.run(100)
    de = dimer.get_potential_energy() - e
    assert abs(de - -0.028) < 0.001
