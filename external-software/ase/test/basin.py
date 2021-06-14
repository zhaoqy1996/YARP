import numpy as np
from ase import Atoms, io
from ase.calculators.lj import LennardJones
from ase.optimize.basin import BasinHopping
from ase.io import read
from ase.units import kB

# Global minima from
# Wales and Doye, J. Phys. Chem. A, vol 101 (1997) 5111-5116
E_global = {
    4: -6.000000,
    5: -9.103852,
    6: -12.712062,
    7: -16.505384}
N = 7
R = N**(1. / 3.)
np.random.seed(42)
pos = np.random.uniform(-R, R, (N, 3))
s = Atoms('He' + str(N),
          positions=pos)
s.set_calculator(LennardJones())
original_positions = 1. * s.get_positions()

ftraj = 'lowest.traj'

for GlobalOptimizer in [BasinHopping(s,
                                     temperature=100 * kB,
                                     dr=0.5,
                                     trajectory=ftraj,
                                     optimizer_logfile=None)]:

    if isinstance(GlobalOptimizer, BasinHopping):
        GlobalOptimizer.run(10)
        Emin, smin = GlobalOptimizer.get_minimum()
    else:
        GlobalOptimizer(totalsteps=10)
        Emin = s.get_potential_energy()
        smin = s
    print("N=", N, 'minimal energy found', Emin,
          ' global minimum:', E_global[N])

    # recalc energy
    smin.set_calculator(LennardJones())
    E = smin.get_potential_energy()
    assert abs(E - Emin) < 1e-15
    smim = read(ftraj)
    E = smin.get_potential_energy()
    assert abs(E - Emin) < 1e-15

    # check that only minima were written
    last_energy = None
    for im in io.read(ftraj + '@:'):
        energy = im.get_potential_energy()
        if last_energy is not None:
            assert energy < last_energy
        last_energy = energy

    # reset positions
    s.set_positions(original_positions)
