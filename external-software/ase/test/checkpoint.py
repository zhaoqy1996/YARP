import os

import numpy as np

from ase import Atom
from ase.build import bulk
from ase.calculators.checkpoint import Checkpoint, CheckpointCalculator
from ase.calculators.lj import LennardJones
from ase.lattice.cubic import Diamond


def op1(a, m):
    a[1].position += m * np.array([0.1, 0.2, 0.3])
    return a


def op2(a, m):
    a += Atom('C', m * np.array([0.2, 0.3, 0.1]))
    return a, a.positions[0]


def test_sqlite():
    print('test_single_file')

    try:
        os.remove('checkpoints.db')
    except OSError:
        pass

    CP = Checkpoint('checkpoints.db')
    a = Diamond('Si', size=[2, 2, 2])
    a = CP(op1)(a, 1.0)
    op1a = a.copy()
    a, ra = CP(op2)(a, 2.0)
    op2a = a.copy()
    op2ra = ra.copy()

    CP = Checkpoint('checkpoints.db')
    a = Diamond('Si', size=[2, 2, 2])
    a = CP(op1)(a, 1.0)
    assert a == op1a
    a, ra = CP(op2)(a, 2.0)
    assert a == op2a
    assert(np.abs(ra - op2ra).max() < 1e-5)


def rattle_calc(atoms, calc):
    try:
        os.remove('checkpoints.db')
    except OSError:
        pass

    orig_atoms = atoms.copy()

    # first do a couple of calculations
    np.random.seed(0)
    atoms.rattle()
    cp_calc_1 = CheckpointCalculator(calc)
    atoms.set_calculator(cp_calc_1)
    e11 = atoms.get_potential_energy()
    f11 = atoms.get_forces()
    atoms.rattle()
    e12 = atoms.get_potential_energy()
    f12 = atoms.get_forces()

    # then re-read them from checkpoint file
    atoms = orig_atoms
    np.random.seed(0)
    atoms.rattle()
    cp_calc_2 = CheckpointCalculator(calc)
    atoms.set_calculator(cp_calc_2)
    e21 = atoms.get_potential_energy()
    f21 = atoms.get_forces()
    atoms.rattle()
    e22 = atoms.get_potential_energy()
    f22 = atoms.get_forces()

    assert e11 == e21
    assert e12 == e22
    assert(np.abs(f11 - f21).max() < 1e-5)
    assert(np.abs(f12 - f22).max() < 1e-5)


def test_new_style_interface():
    calc = LennardJones()
    atoms = bulk('Cu')
    rattle_calc(atoms, calc)


test_sqlite()
test_new_style_interface()
