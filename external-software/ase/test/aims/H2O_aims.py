from ase import Atoms
from ase.calculators.aims import Aims, AimsCube
from ase.optimize import QuasiNewton

water = Atoms('HOH', [(1, 0, 0), (0, 0, 0), (0, 1, 0)])

water_cube = AimsCube(points=(29, 29, 29),
                      plots=('total_density',
                             'delta_density',
                             'eigenstate 5',
                             'eigenstate 6'))

calc = Aims(xc='PBE',
            output=['dipole'],
            sc_accuracy_etot=1e-6,
            sc_accuracy_eev=1e-3,
            sc_accuracy_rho=1e-6,
            sc_accuracy_forces=1e-4,
            cubes=water_cube)

water.set_calculator(calc)
dynamics = QuasiNewton(water, trajectory='square_water.traj')
dynamics.run(fmax=0.01)
