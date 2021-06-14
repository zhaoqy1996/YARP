from math import radians, sin, cos

from ase import Atoms
from ase.neb import NEB
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton, BFGS
from ase.visualize import view

from ase.calculators.turbomole import Turbomole

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
              positions=[(- sin(angle) * doht, 0., cos(angle) * doht),
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

# Write all commands for the define command in a string
define_str = ('\n\na coord\n\n*\nno\nb all 3-21g '
              'hondo\n*\neht\n\n-1\nno\ns\n*\n\ndft\non\nfunc '
              'pwlda\n\n\nscf\niter\n300\n\n*')

# Set constraints and calculator:
constraint = FixAtoms(indices=[1, 3])  # fix OO BUG No.1: fixes atom 0 and 1
# constraint = FixAtoms(mask=[0,1,0,1,0]) # fix OO    #Works without patch
for image in images:
    image.set_calculator(Turbomole(define_str=define_str))
    image.set_constraint(constraint)

# Relax initial and final states:
if 1:
    dyn1 = QuasiNewton(images[0])
    dyn1.run(fmax=0.10)
    dyn2 = QuasiNewton(images[-1])
    dyn2.run(fmax=0.10)


# Interpolate positions between initial and final states:
neb.interpolate()
if 1:
    for image in images:
        print(image.get_distance(1, 2), image.get_potential_energy())

dyn = BFGS(neb, trajectory='turbomole_h3o2m.traj')
dyn.run(fmax=0.10)

for image in images:
    print(image.get_distance(1, 2), image.get_potential_energy())
