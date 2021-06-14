import ase.calculators.demon as demon
from ase import Atoms
#from ase.optimize import BFGS
import numpy as np

# d = 0.9575
d = 0.9775
# t = np.pi / 180 * 104.51
t = np.pi / 180 * 110.51
atoms = Atoms('H2O',
              positions=[(d, 0, 0),
                         (d * np.cos(t), d * np.sin(t), 0),
                         (0, 0, 0)])

# set up deMon calculator
basis = {'all': 'aug-cc-pvdz'}
auxis = {'all': 'GEN-A2*'}


# XAS hch
input_arguments = {'GRID': 'FINE',
                   'MOMODIFY': [[1,0],
                                [1,0.5]], 
                   'CHARGE':0,
                   'XRAY':'XAS'}

calc = demon.Demon(basis=basis,
                   auxis=auxis,
                   scftype='UKS TOL=1.0E-6 CDF=1.0E-5',
                   guess='TB',
                   xc=['BLYP', 'BASIS'],
                   input_arguments=input_arguments)

atoms.set_calculator(calc)

# energy
print('XAS hch')
print('energy')
energy = atoms.get_potential_energy()
print(energy)
ref = -1815.44708987 #-469.604737006
error = np.sqrt(np.sum((energy - ref)**2))
print('diff from reference:')
print(error)

tol = 1.0e-4
assert(error < tol)

# check xas
results = calc.results

print('xray, first transition, energy')
value =results['xray']['E_trans'][0]
print(value)
ref = 539.410015646
error = np.sqrt(np.sum((value- ref)**2))
print('diff from reference:')
print(error)

tol = 1.0e-4
assert(error < tol)

print('xray, first transition, transition dipole moments')
value = results['xray']['trans_dip'][0]
print(value)
ref = np.array([1.11921906e-02, 1.61393975e-02, 1.70983631e-07])
error = np.sqrt(np.sum((value- ref)**2))
print('diff from reference:')
print(error)

tol = 1.0e-4
assert(error < tol)


# XES
input_arguments = {'GRID': 'FINE',
                   'CHARGE':0,
                   'XRAY':'XES ALPHA=1-1'}

calc = demon.Demon(basis=basis,
                   auxis=auxis,
                   scftype='UKS TOL=1.0E-6 CDF=1.0E-5',
                   guess='TB',
                   xc=['BLYP', 'BASIS'],
                   input_arguments=input_arguments)

atoms.set_calculator(calc)

# energy
print('')
print('XES')
print('energy')
energy = atoms.get_potential_energy()
print(energy)
ref = -2079.6635944 
error = np.sqrt(np.sum((energy - ref)**2))
print('diff from reference:')
print(error)

tol = 1.0e-4
assert(error < tol)

# check xes
results = calc.results

print('xray, first transition, energy')
value =results['xray']['E_trans'][0]
print(value)
ref = 486.862715888 #539.410015646
error = np.sqrt(np.sum((value- ref)**2))
print('diff from reference:')
print(error)

tol = 1.0e-4
assert(error < tol)

print('xray, first transition, transition dipole moments')
value = results['xray']['trans_dip'][0]
print(value)
ref = np.array([6.50528073e-03, 9.37895253e-03, 6.99433480e-09])
error = np.sqrt(np.sum((value- ref)**2))
print('diff from reference:')
print(error)

tol = 1.0e-4
assert(error < tol)

# and XPS
input_arguments = {'GRID': 'FINE',
                   'MOMODIFY': [[1,0],
                                [1,0.0]], 
                   'CHARGE':0,
                   'XRAY':'XAS'}

calc = demon.Demon(basis=basis,
                   auxis=auxis,
                   scftype='UKS TOL=1.0E-6 CDF=1.0E-5',
                   guess='TB',
                   xc=['BLYP', 'BASIS'],
                   input_arguments=input_arguments)

atoms.set_calculator(calc)


# energy
print('')
print('XPS')
print('energy')
energy = atoms.get_potential_energy()
print(energy)
ref = -1536.9295935
error = np.sqrt(np.sum((energy - ref)**2))
print('diff from reference:')
print(error)

tol = 1.0e-4
assert(error < tol)

# First excited state
input_arguments = {'GRID': 'FINE',
                   'MOMODIFY': [[1,0],
                                [1,0.0]], 
                   'CHARGE':-1}

calc = demon.Demon(basis=basis,
                   auxis=auxis,
                   scftype='UKS TOL=1.0E-6 CDF=1.0E-5',
                   guess='TB',
                   xc=['BLYP', 'BASIS'],
                   input_arguments=input_arguments)

atoms.set_calculator(calc)


# energy
print('')
print('EXC')
print('energy')
energy = atoms.get_potential_energy()
print(energy)
ref = -1543.18092135
error = np.sqrt(np.sum((energy - ref)**2))
print('diff from reference:')
print(error)

tol = 1.0e-4
assert(error < tol)


print('tests passed')


