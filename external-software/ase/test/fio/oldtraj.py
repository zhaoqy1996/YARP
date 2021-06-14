"""Make sure we can't read old traj file, but we can convert them."""
import ase.io as aio
from ase import Atoms
from ase.io.pickletrajectory import PickleTrajectory
from ase.io.trajectory import convert
from ase.test import must_raise


t = PickleTrajectory('hmm.traj', 'w', _warn=False)
a = Atoms('H')
t.write(a)
t.close()

with must_raise(DeprecationWarning):
    aio.read('hmm.traj')
    
convert('hmm.traj')
aio.read('hmm.traj')
