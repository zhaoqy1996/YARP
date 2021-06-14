from ase.calculators.lj import LennardJones
from ase.optimize import FIRE, BFGS
from ase.neb import NEB, NEBTools
from ase import Atoms

nimages = 3
fmax = 0.01

for remove_rotation_and_translation in [True, False]:
    # Define coordinates for initial and final states
    initial = Atoms('O4',
                    [(1.94366484, 2.24788196, 2.32204726),
                     (3.05353823, 2.08091038, 2.30712548),
                     (2.63770601, 3.05694348, 2.67368242),
                     (2.50579418, 2.12540646, 3.28585811)])

    final = Atoms('O4',
                  [(1.95501370, 2.22270649, 2.33191017),
                   (3.07439495, 2.13662682, 2.31948449),
                   (2.44730550, 1.26930465, 2.65964947),
                   (2.52788189, 2.18990240, 3.29728667)])

    final.set_cell((5, 5, 5))
    initial.set_cell((5, 5, 5))
    final.set_calculator(LennardJones())
    initial.set_calculator(LennardJones())

    images = [initial]

    # Set calculator
    for i in range(nimages):
        image = initial.copy()
        image.set_calculator(LennardJones())
        images.append(image)

    images.append(final)

    # Define the NEB and make a linear interpolation
    # with removing translational
    # and rotational degrees of freedom
    neb = NEB(images,
              remove_rotation_and_translation=remove_rotation_and_translation)
    neb.interpolate()
    # Test used these old defaults which are not optimial, but work
    # in this particular system
    neb.idpp_interpolate(fmax=0.1, optimizer=BFGS)

    qn = FIRE(neb, dt=0.005, maxmove=0.05, dtmax=0.1)
    qn.run(steps=20)

    # Switch to CI-NEB, still removing the external degrees of freedom
    # Also spesify the linearly varying spring constants
    neb = NEB(images, climb=True,
              remove_rotation_and_translation=remove_rotation_and_translation)
    qn = FIRE(neb, dt=0.005, maxmove=0.05, dtmax=0.1)
    qn.run(fmax=fmax)

    images = neb.images

    nebtools = NEBTools(images)
    Ef_neb, dE_neb = nebtools.get_barrier(fit=False)
    nsteps_neb = qn.nsteps
    if remove_rotation_and_translation:
        Ef_neb_0 = Ef_neb
        nsteps_neb_0 = nsteps_neb

assert abs(Ef_neb - Ef_neb_0) < 1e-2
assert nsteps_neb_0 < nsteps_neb * 0.7
