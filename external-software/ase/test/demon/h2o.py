import ase.calculators.demon as demon
from ase import Atoms
from ase.optimize import BFGS
import numpy as np

tol = 1.0e-6

# d = 0.9575
d = 0.9775
# t = np.pi / 180 * 104.51
t = np.pi / 180 * 110.51
atoms = Atoms('H2O',
              positions=[(d, 0, 0),
                         (d * np.cos(t), d * np.sin(t), 0),
                         (0, 0, 0)])

# set up deMon calculator
basis = {'all': 'aug-cc-pvdz',
         'O': 'RECP6|SD'}
auxis = {'all': 'GEN-A2*'}
input_arguments = {'GRID': 'FINE'}
    
calc = demon.Demon(basis=basis,
                   auxis=auxis,
                   scftype='RKS TOL=1.0E-6 CDF=1.0E-5',
                   guess='TB',
                   xc=['BLYP', 'BASIS'],
                   input_arguments=input_arguments)

atoms.set_calculator(calc)

# energy
energy = atoms.get_potential_energy()

ref = -469.604737006
print('energy')
print(energy)
error = np.sqrt(np.sum((energy - ref)**2))
print('diff from reference:')
print(error)

tol = 1.0e-6
assert(error < tol)

# dipole
dipole = atoms.get_dipole_moment()

ref = np.array([0.19228183, 0.27726241, 0.0])
error = np.sqrt(np.sum((dipole - ref)**2))
print('dipole')
print(dipole)
print('diff from reference:')
print(error)

tol = 1.0e-4
assert(error < tol)


# numerical forces
forces_num = calc.calculate_numerical_forces(atoms, d=0.001)

ref = np.array([[-1.26056746e-01, 4.10007559e-01, 2.85719551e-04],
                [4.28062314e-01, 2.56059142e-02, 2.17691110e-04],
                [-3.02019173e-01, -4.35613473e-01, -5.03410632e-04]])

error = np.sqrt(np.sum((forces_num - ref)**2))
print('forces_num')
print(forces_num)
print('diff from reference:')
print(error)

tol = 1.0e-4
assert(error < tol)


# analytical forces
forces_an = atoms.get_forces()

ref = np.array([[-1.26446863e-01, 4.09628186e-01, -0.00000000e+00],
                [4.27934442e-01, 2.50425467e-02, -5.14220671e-05],
                [-2.99225008e-01, -4.31533987e-01, -5.14220671e-05]])

error = np.sqrt(np.sum((forces_an - ref)**2))
print('forces_an')
print(forces_an)
print('diff from reference:')
print(error)

tol = 1.0e-3
assert(error < tol)

# optimize geometry
dyn = BFGS(atoms)
dyn.run(fmax=0.01)

positions = atoms.get_positions()

ref = np.array([[  9.61364579e-01, 2.81689367e-02, -1.58730770e-06],
                [ -3.10444398e-01, 9.10289261e-01, -5.66399075e-06],
                [ -1.56957763e-02, -2.26044053e-02, -2.34155615e-06]])

error = np.sqrt(np.sum((positions - ref)**2))
print('positions')
print(positions)
print('diff from reference:')
print(error)

tol = 1.0e-3
assert(error < tol)

print('tests passed')



