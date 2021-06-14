from math import pi
from ase.calculators.emt import EMT
from ase.constraints import FixInternals
from ase.optimize.bfgs import BFGS
from ase.build import molecule


system = molecule('CH3CH2OH', vacuum=5.0)
system.rattle(stdev=0.3)

# Angles, Bonds, Dihedrals are built up with  pairs of constraint
# value and indices defining the constraint

# Fix this dihedral angle to whatever it was from the start
indices = [6, 0, 1, 2]
dihedral1 = system.get_dihedral(*indices)

# Fix angle to whatever it was from the start
indices2 = [6, 0, 1]
angle1 = system.get_angle(*indices2)
# system.set_dihedral(indices, pi/20, mask=[0,1,1,1,1,1,0,0,0])

# Fix bond between atoms 1 and 2 to 1.4
target_bondlength = 1.4
indices_bondlength = [1, 2]

constraint = FixInternals(bonds=[(target_bondlength, indices_bondlength)],
                          angles=[(angle1 * pi / 180, indices2)],
                          dihedrals=[(dihedral1 * pi / 180, indices)],
                          epsilon=1e-10)

print(constraint)

calc = EMT()

opt = BFGS(system, trajectory='opt.traj', logfile='opt.log')

previous_angle = system.get_angle(*indices2)
previous_dihedral = system.get_dihedral(*indices)

print('angle before', previous_angle)
print('dihedral before', previous_dihedral)
print('bond length before', system.get_distance(*indices_bondlength))
print('(target bondlength %s)', target_bondlength)

system.set_calculator(calc)
system.set_constraint(constraint)
print('-----Optimization-----')
opt.run(fmax=0.01)

new_angle = system.get_angle(*indices2)
new_dihedral = system.get_dihedral(*indices)
new_bondlength = system.get_distance(*indices_bondlength)

print('angle after', new_angle)
print('dihedral after', new_dihedral)
print('bondlength after', new_bondlength)

err1 = new_angle - previous_angle
err2 = new_dihedral - previous_dihedral
err3 = new_bondlength - target_bondlength

print('error in angle', repr(err1))
print('error in dihedral', repr(err2))
print('error in bondlength', repr(err3))

assert err1 < 1e-11
assert err2 < 1e-12
assert err3 < 1e-12
