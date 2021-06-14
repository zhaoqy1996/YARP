import numpy as np

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize.precon import Exp, PreconLBFGS, PreconFIRE
from ase.constraints import FixBondLength, FixAtoms

N = 1
a0 = bulk('Cu', cubic=True)
a0 *= (N, N, N)

# perturb the atoms
s = a0.get_scaled_positions()
s[:, 0] *= 0.995
a0.set_scaled_positions(s)

nsteps = []
energies = []
for OPT in [PreconLBFGS, PreconFIRE]:
    for precon in [None, Exp(A=3, mu=1.0)]:
        atoms = a0.copy()
        atoms.set_calculator(EMT())
        opt = OPT(atoms, precon=precon, use_armijo=True)
        opt.run(1e-4)
        energies += [atoms.get_potential_energy()]
        nsteps += [opt.get_number_of_steps()]

# check we get the expected energy for all methods
assert np.abs(np.array(energies) - -0.022726045433998365).max() < 1e-4

# test with fixed bondlength and fixed atom constraints
cu0 = bulk("Cu") * (5, 5, 5)
cu0.rattle(0.01)
a0 = cu0.get_distance(0, 1)
cons = [FixBondLength(0,1), FixAtoms([2,3])]
for precon in [None, Exp(mu=1.0)]:
    cu = cu0.copy()
    cu.set_calculator(EMT())
    cu.set_distance(0, 1, a0*1.2)
    cu.set_constraint(cons)
    opt = PreconLBFGS(cu, precon=precon, use_armijo=True)
    opt.run(fmax=1e-3)

    assert abs(cu.get_distance(0, 1)/a0 - 1.2) < 1e-3
    assert np.all(abs(cu.positions[2] - cu0.positions[2]) < 1e-3)
    assert np.all(abs(cu.positions[3] - cu0.positions[3]) < 1e-3)
