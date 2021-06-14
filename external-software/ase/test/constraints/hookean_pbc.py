from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import Hookean

L = 8.  # length of the cubic box
d = 2.3  # Au-Au distance 
cell = [L]*3
positions = [[(L - d/2) % L , L/2, L/2], [(L + d/2) % L, L/2, L/2]]
a = Atoms('AuAu', cell=[L]*3, positions=positions, pbc=True)

a.set_calculator(EMT())
e1 = a.get_potential_energy()

constraint = Hookean(a1=0, a2=1, rt=1.1*d, k=10.)
a.set_constraint(constraint)
e2 = a.get_potential_energy()

a.set_pbc([False, True, True])
e3 = a.get_potential_energy()

assert abs(e1 - e2) < 1e-8
assert not abs(e1 - e3) < 1e-8
