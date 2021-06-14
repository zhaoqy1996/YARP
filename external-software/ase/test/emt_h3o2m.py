from math import radians, sin, cos

from ase import Atoms
from ase.neb import NEB
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton, BFGS
from ase.visualize import view

# http://jcp.aip.org/resource/1/jcpsa6/v97/i10/p7507_s1
doo = 2.74
doht = 0.957
doh = 0.977
angle = radians(104.5)
initial = Atoms('HOHOH',
                positions=[(-sin(angle) * doht, 0., cos(angle) * doht),
                           (0., 0., 0.),
                           (0., 0., doh),
                           (0., 0., doo),
                           (sin(angle) * doht, 0., doo - cos(angle) * doht)])
if 0:
    view(initial)

final = Atoms('HOHOH',
              positions=[(-sin(angle) * doht, 0., cos(angle) * doht),
                         (0., 0., 0.),
                         (0., 0., doo - doh),
                         (0., 0., doo),
                         (sin(angle) * doht, 0., doo - cos(angle) * doht)])
if 0:
    view(final)

# Make band:
images = [initial.copy()]
for i in range(3):
    images.append(initial.copy())
images.append(final.copy())
neb = NEB(images, climb=True)

# Set constraints and calculator:
constraint = FixAtoms(indices=[1, 3])  # fix OO
for image in images:
    image.set_calculator(EMT())
    image.set_constraint(constraint)

for image in images:  # O-H(shared) distance
    print(image.get_distance(1, 2), image.get_potential_energy())

# Relax initial and final states:
if 1:
    # XXX: Warning:
    # One would have to optimize more tightly in order to get
    # symmetric anion from both images[0] and [1], but
    # if one optimizes tightly one gets rotated(H2O) ... OH- instead
    dyn1 = QuasiNewton(images[0])
    dyn1.run(fmax=0.01)
    dyn2 = QuasiNewton(images[-1])
    dyn2.run(fmax=0.01)

# Interpolate positions between initial and final states:
neb.interpolate()

for image in images:
    print(image.get_distance(1, 2), image.get_potential_energy())

dyn = BFGS(neb, trajectory='emt_h3o2m.traj')
dyn.run(fmax=0.05)

for image in images:
    print(image.get_distance(1, 2), image.get_potential_energy())
