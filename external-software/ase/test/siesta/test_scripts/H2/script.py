from ase.calculators.siesta import Siesta
from ase.calculators.siesta.parameters import Species, PAOBasisBlock
from ase.optimize import QuasiNewton
from ase import Atoms

atoms = Atoms(
    '3H',
    [(0.0, 0.0, 0.0),
     (0.0, 0.0, 0.5),
     (0.0, 0.0, 1.0)],
    cell=[10, 10, 10])

basis_set = PAOBasisBlock(
    """1
0  2 S 0.2
0.0 0.0""")
atoms.set_tags([0, 1, 0])
siesta = Siesta(
    species=[
        Species(symbol='H', tag=None, basis_set='SZ'),
        Species(symbol='H', tag=1, basis_set=basis_set, ghost=True)])

atoms.set_calculator(siesta)
dyn = QuasiNewton(atoms, trajectory='h.traj')
dyn.run(fmax=0.02)
