import os
import warnings

import numpy as np

from ase import Atom, Atoms
from ase.io import read
from ase.io import NetCDFTrajectory

co = Atoms([Atom('C', (0, 0, 0)),
            Atom('O', (0, 0, 1.2))],
            cell=[3, 3, 3],
            pbc=True)
traj = NetCDFTrajectory('1.nc', 'w', co)
for i in range(5):
    co.positions[:, 2] += 0.1
    traj.write()
del traj
traj = NetCDFTrajectory('1.nc', 'a')
co = traj[-1]
print(co.positions)
co.positions[:] += 1
traj.write(co)
del traj
t = NetCDFTrajectory('1.nc', 'a')

print(t[-1].positions)
print('.--------')
for i, a in enumerate(t):
    if i < 4:
        print(1, a.positions[-1, 2], 1.3 + i * 0.1)
        assert abs(a.positions[-1, 2] - 1.3 - i * 0.1) < 1e-6
        assert a.pbc.all()
    else:
        print(1, a.positions[-1, 2], 1.7 + i - 4)
        assert abs(a.positions[-1, 2] - 1.7 - i + 4) < 1e-6
        assert a.pbc.all()
co.positions[:] += 1
t.write(co)
for i, a in enumerate(t):
    if i < 4:
        print(2, a.positions[-1, 2], 1.3 + i * 0.1)
        assert abs(a.positions[-1, 2] - 1.3 - i * 0.1) < 1e-6
    else:
        print(2, a.positions[-1, 2], 1.7 + i - 4)
        assert abs(a.positions[-1, 2] - 1.7 - i + 4) < 1e-6
assert len(t) == 7

# Change atom type and append
co[0].number = 1
t.write(co)
t2 = NetCDFTrajectory('1.nc', 'r')
co2 = t2[-1]
assert (co2.numbers == co.numbers).all()
del t2

co[0].number = 6
co.pbc = True
t.write(co)

co.pbc = False
o = co.pop(1)
try:
    t.write(co)
except ValueError:
    pass
else:
    assert False

co.append(o)
co.pbc = True
t.write(co)
del t

# append to a nonexisting file
fname = '2.nc'
if os.path.isfile(fname):
    os.remove(fname)
t = NetCDFTrajectory(fname, 'a', co)
del t

fname = '3.nc'
t = NetCDFTrajectory(fname, 'w', co)
# File is not created before first write
co.set_pbc([True, False, False])
d = co.get_distance(0, 1)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    t.write(co)
del t
# Check pbc
for c in [1, 1000]:
    t = NetCDFTrajectory(fname, chunk_size=c)
    a = t[-1]
    assert a.pbc[0] and not a.pbc[1] and not a.pbc[2]
    assert abs(a.get_distance(0, 1) - d) < 1e-6
    del t
# Append something in Voigt notation
t = NetCDFTrajectory(fname, 'a')
for frame, a in enumerate(t):
    test = np.random.random([len(a), 6])
    a.set_array('test', test)
    t.write_arrays(a, frame, ['test'])
del t
os.remove(fname)

# Check cell origin
co.set_pbc(True)
co.set_celldisp([1,2,3])
traj = NetCDFTrajectory('4.nc', 'w', co)
traj.write(co)
traj.close()

traj = NetCDFTrajectory('4.nc', 'r')
a = traj[0]
assert np.all(abs(a.get_celldisp() - np.array([1,2,3])) < 1e-12)
traj.close()

os.remove('4.nc')

# Add 'id' field and check if it is read correctly
co.set_array('id', np.array([2, 1]))
traj = NetCDFTrajectory('5.nc', 'w', co)
traj.write(co, arrays=['id'])
traj.close()

traj = NetCDFTrajectory('5.nc', 'r')#
assert np.all(traj[0].numbers == [8, 6])
assert np.all(np.abs(traj[0].positions - np.array([[2, 2, 3.7], [2., 2., 2.5]])) < 1e-6)
traj.close()

a = read('5.nc')
assert(len(a) == 2)

os.remove('5.nc')
