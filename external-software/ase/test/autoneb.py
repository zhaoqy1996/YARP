from ase.build import fcc211, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
from ase.neb import NEBTools
from ase.autoneb import AutoNEB

# Pt atom adsorbed in a hollow site:
slab = fcc211('Pt', size=(3, 2, 2), vacuum=4.0)
add_adsorbate(slab, 'Pt', 0.5, (-0.1, 2.7))

# Fix second and third layers:
slab.set_constraint(FixAtoms(range(6, 12)))

# Use EMT potential:
slab.set_calculator(EMT())

# Initial state:
qn = QuasiNewton(slab, trajectory='neb000.traj')
qn.run(fmax=0.05)

# Final state:
slab[-1].x += slab.get_cell()[0, 0]
slab[-1].y += 2.8
qn = QuasiNewton(slab, trajectory='neb001.traj')
qn.run(fmax=0.05)

# Stops PermissionError on Win32 for access to
# the traj file that remains open.
del qn


def attach_calculators(images):
    for i in range(len(images)):
        images[i].set_calculator(EMT())


autoneb = AutoNEB(attach_calculators,
                  prefix='neb',
                  optimizer='BFGS',
                  n_simul=3,
                  n_max=7,
                  fmax=0.05,
                  k=0.5,
                  parallel=False,
                  maxsteps=[50, 1000])
autoneb.run()

nebtools = NEBTools(autoneb.all_images)
assert abs(nebtools.get_barrier()[0] - 0.938) < 1e-3
