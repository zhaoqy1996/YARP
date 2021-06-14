import numpy as np
from ase import Atoms
from ase.units import fs, kB
from ase.calculators.test import TestPotential
from ase.md import Langevin
from ase.io import Trajectory, read
from ase.optimize import QuasiNewton
from ase.utils import seterr

rng = np.random.RandomState(0)

with seterr(all='raise'):
    a = Atoms('4X',
              masses=[1, 2, 3, 4],
              positions=[(0, 0, 0),
                         (1, 0, 0),
                         (0, 1, 0),
                         (0.1, 0.2, 0.7)],
              calculator=TestPotential())
    print(a.get_forces())
    # Langevin should reproduce Verlet if friction is 0.
    md = Langevin(a, 0.5 * fs, 300 * kB, 0.0, logfile='-', loginterval=500)
    traj = Trajectory('4N.traj', 'w', a)
    md.attach(traj, 100)
    e0 = a.get_total_energy()
    md.run(steps=10000)
    del traj
    assert abs(read('4N.traj').get_total_energy() - e0) < 0.0001

    # Try again with nonzero friction.
    md = Langevin(a, 0.5 * fs, 300 * kB, 0.001, logfile='-', loginterval=500,
                  rng=rng)
    traj = Trajectory('4NA.traj', 'w', a)
    md.attach(traj, 100)
    md.run(steps=10000)

    # We cannot test the temperature without a lot of statistics.
    # Asap does that.  But if temperature is quite unreasonable,
    # something is very wrong.
    T = a.get_temperature()
    assert T > 50
    assert T < 1000

    qn = QuasiNewton(a)
    qn.run(0.001)
    assert abs(a.get_potential_energy() - 1.0) < 0.000002
