from ase.test import must_raise

import os
from ase import Atom, Atoms
from ase.io import Trajectory, read
from ase.constraints import FixBondLength
from ase.calculators.calculator import PropertyNotImplementedError

co = Atoms([Atom('C', (0, 0, 0)),
            Atom('O', (0, 0, 1.2))])
traj = Trajectory('1.traj', 'w', co)

written = []

for i in range(5):
    co.positions[:, 2] += 0.1
    traj.write()
    written.append(co.copy())

traj = Trajectory('1.traj', 'a')
co = read('1.traj')
print(co.positions)
co.positions[:] += 1
traj.write(co)
written.append(co.copy())

for a in Trajectory('1.traj'):
    print(1, a.positions[-1, 2])
co.positions[:] += 1
t = Trajectory('1.traj', 'a')
t.write(co)
written.append(co.copy())
assert len(t) == 7

co[0].number = 1
t.write(co)
written.append(co.copy())

co[0].number = 6
co.pbc = True
t.write(co)
written.append(co.copy())

co.pbc = False
o = co.pop(1)
t.write(co)
written.append(co.copy())

co.append(o)
t.write(co)
written.append(co.copy())

imgs = read('1.traj@:')
assert len(imgs) == len(written)
for img1, img2 in zip(imgs, written):
    assert img1 == img2

# append to a nonexisting file:
fname = '2.traj'
if os.path.isfile(fname):
    os.remove(fname)
t = Trajectory(fname, 'a', co)
t.close()
os.remove(fname)

t = Trajectory('empty.traj', 'w')
t.close()
t = Trajectory('empty.traj', 'r')
assert len(t) == 0

t = Trajectory('fake.traj', 'w')
t.write(Atoms('H'), energy=-42.0, forces=[[1, 2, 3]])

t = Trajectory('only-energy.traj', 'w', properties=['energy'])
a = read('fake.traj')
t.write(a)
b = read('only-energy.traj')
e = b.get_potential_energy()
assert e + 42 == 0
with must_raise(PropertyNotImplementedError):
    f = b.get_forces()

# Make sure constraints play well with momenta:
a = Atoms('H2',
          positions=[(0, 0, 0), (0, 0, 1)],
          momenta=[(1, 0, 0), (0, 0, 0)])
a.constraints = [FixBondLength(0, 1)]
t = Trajectory('constraint.traj', 'w', a)
t.write()
b = read('constraint.traj')
assert not (b.get_momenta() - a.get_momenta()).any()
