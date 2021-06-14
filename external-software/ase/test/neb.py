from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import Trajectory, read
from ase.neb import NEB, NEBTools
from ase.calculators.morse import MorsePotential
from ase.optimize import BFGS, QuasiNewton

atoms = Atoms('H7',
              positions=[(0, 0, 0),
                         (1, 0, 0),
                         (0, 1, 0),
                         (1, 1, 0),
                         (0, 2, 0),
                         (1, 2, 0),
                         (0.5, 0.5, 1)],
              constraint=[FixAtoms(range(6))],
              calculator=MorsePotential())

traj = Trajectory('H.traj', 'w', atoms)
dyn = QuasiNewton(atoms, maxstep=0.2)
dyn.attach(traj.write)
dyn.run(fmax=0.01, steps=100)

print(atoms)
del atoms[-1]
print(atoms)
del atoms[5]
print(atoms)
assert len(atoms.constraints[0].index) == 5

fmax = 0.05
nimages = 3

print([a.get_potential_energy() for a in Trajectory('H.traj')])
images = [Trajectory('H.traj')[-1]]
for i in range(nimages):
    images.append(images[0].copy())
images[-1].positions[6, 1] = 2 - images[0].positions[6, 1]
neb = NEB(images)
neb.interpolate()
if 0:  # verify that initial images make sense
    from ase.visualize import view
    view(neb.images)

for image in images:
    image.set_calculator(MorsePotential())

dyn = BFGS(neb, trajectory='mep.traj')  # , logfile='mep.log')

dyn.run(fmax=fmax)

for a in neb.images:
    print(a.positions[-1], a.get_potential_energy())

neb.climb = True
dyn.run(fmax=fmax)

# Check NEB tools.
nt_images = read('mep.traj@-4:')
nebtools = NEBTools(nt_images)
nt_fmax = nebtools.get_fmax(climb=True)
Ef, dE = nebtools.get_barrier()
print(Ef, dE, fmax, nt_fmax)
assert nt_fmax < fmax
assert abs(Ef - 1.389) < 0.001
