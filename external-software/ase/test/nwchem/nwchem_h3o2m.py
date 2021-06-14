from math import radians, sin, cos

from ase import Atoms
from ase.neb import NEB
from ase.constraints import FixAtoms
from ase.calculators.nwchem import NWChem
from ase.optimize import QuasiNewton, BFGS
from ase.visualize import view

# http://jcp.aip.org/resource/1/jcpsa6/v97/i10/p7507_s1
doo = 2.74
doht = 0.957
doh = 0.977
angle = radians(104.5)
initial = Atoms('HOHOH',
                positions=[(-sin(angle) * doht, 0, cos(angle) * doht),
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

def calculator():
    return NWChem(task='gradient',
                  geometry='nocenter noautosym noautoz',
                  charge=-1)

# Set constraints and calculator:
constraint = FixAtoms(indices=[1, 3])  # fix OO
for image in images:
    image.set_calculator(calculator())
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

dyn = BFGS(neb, trajectory='nwchem_h3o2m.traj')
dyn.run(fmax=0.10)  # use better basis (e.g. aug-cc-pvdz) for NEB to converge

for image in images:
    print(image.get_distance(1, 2), image.get_potential_energy())
