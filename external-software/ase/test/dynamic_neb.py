import numpy as np
from ase import Atoms
from ase.build import fcc111
from ase.optimize import BFGS
from ase.calculators.emt import EMT as OrigEMT
from ase.neb import NEB

# Global counter of force evaluations:
force_evaluations = [0]

class EMT(OrigEMT):
    def calculate(self, *args, **kwargs):
        force_evaluations[0] += 1
        OrigEMT.calculate(self, *args, **kwargs)


# Build Pt(111) slab with six surface atoms and add oxygen adsorbate
initial = fcc111('Pt', size=(3, 2, 3), orthogonal=True)
initial.center(axis=2, vacuum=10)
oxygen = Atoms('O')
oxygen.translate(initial[7].position + (0., 0., 3.5))
initial.extend(oxygen)

# EMT potential
calc = EMT()
initial.set_calculator(EMT())

# Optimize initial state
opt = BFGS(initial)
opt.run(fmax=0.03)

# Move oxygen adsorbate to neighboring hollow site
final = initial.copy()
final[18].x += 2.8
final[18].y += 1.8

final.set_calculator(EMT())

opt = BFGS(final)
opt.run(fmax=0.03)

# NEB with five interior images
images = [initial]
for i in range(5):
    images.append(initial.copy())
images.append(final)

fmax = 0.03  # Same for NEB and optimizer

for i in range(1, len(images)-1):
    calc = EMT()
    images[i].set_calculator(calc)

# Dynamic NEB
neb = NEB(images, fmax=fmax, dynamic_relaxation=True)
neb.interpolate()

# Optimize and check number of calculations with dynamic NEB.
# We use a hack with a global counter to count the force evaluations:
force_evaluations[0] = 0
opt = BFGS(neb)
opt.run(fmax=fmax)
ncalculations_dyn = force_evaluations[0]

# Get potential energy of transition state
Emax_dyn = np.sort([image.get_potential_energy()
                    for image in images[1:-1]])[-1]

# Default NEB
neb = NEB(images, dynamic_relaxation=False)
neb.interpolate()

# Optimize and check number of calculations for default NEB:
force_evaluations[0] = 0
opt = BFGS(neb)
opt.run(fmax=fmax)
ncalculations_default = force_evaluations[0]

# Get potential energy of transition state
Emax_def = np.sort([image.get_potential_energy()
                    for image in images[1:-1]])[-1]

# Check force calculation count for default and dynamic NEB implementations
print(ncalculations_dyn, ncalculations_default)
assert ncalculations_dyn < ncalculations_default

# Assert reaction barriers are within 1 meV of each other
assert(abs(Emax_dyn - Emax_def) < 1e-3)
