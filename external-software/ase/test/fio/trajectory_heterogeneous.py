from __future__ import print_function
from ase.constraints import FixAtoms, FixBondLength
from ase.build import molecule, bulk
from ase.io.trajectory import Trajectory, get_header_data
from ase.io import read

a0 = molecule('H2O')
a1 = a0.copy()
a1.rattle(stdev=0.5)
a2 = a0.copy()
a2.set_masses()
a2.center(vacuum=2.0)
a2.rattle(stdev=0.2)
a3 = molecule('CH3CH2OH')
a4 = bulk('Au').repeat((2, 2, 2))
a5 = bulk('Cu').repeat((2, 2, 3))

# Add constraints to some of the images:
images = [a0, a1, a2, a3, a4, a5]
for i, img in enumerate(images[3:]):
    img.set_constraint(FixAtoms(indices=range(i + 3)))
    if i == 2:
        img.constraints.append(FixBondLength(5, 6))

traj = Trajectory('out.traj', 'w')
for i, img in enumerate(images):
    traj.write(img)
    print(i, traj.multiple_headers)
    assert traj.multiple_headers == (i >= 2)
traj.close()

rtraj = Trajectory('out.traj')
newimages = list(rtraj)

assert len(images) == len(newimages)
for i in range(len(images)):
    assert images[i] == newimages[i], i
    h1 = get_header_data(images[i])
    h2 = get_header_data(newimages[i])
    print(i, images[i])
    print(h1)
    print(h2)
    print()
    # assert headers_equal(h1, h2)

# Test append mode:
with Trajectory('out.traj', 'a') as atraj:
    atraj.write(molecule('H2'))
    atraj.write(molecule('H2'))
read('out.traj@:')
