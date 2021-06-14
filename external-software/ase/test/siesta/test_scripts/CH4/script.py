from ase.units import Ry, eV
from ase.io import read
from ase.calculators.siesta.parameters import Species, PAOBasisBlock
from ase.calculators.siesta.siesta import Siesta
from ase.optimize import QuasiNewton
from ase import Atoms
import numpy as np

traj = 'bud.traj'

try:
    bud = read(traj)
except:
    bud = Atoms('CH4', np.array([
        [0.000000, 0.000000, 0.100000],
        [0.682793, 0.682793, 0.682793],
        [-0.682793, -0.682793, 0.68279],
        [-0.682793, 0.682793, -0.682793],
        [0.682793, -0.682793, -0.682793]]),
        cell=[10, 10, 10])

c_basis = """2 nodes 1.00
0 1 S 0.20 P 1 0.20 6.00
5.00
1.00
1 2 S 0.20 P 1 E 0.20 6.00
6.00 5.00
1.00 0.95"""

species = Species(symbol='C', basis_set=PAOBasisBlock(c_basis))
calc = Siesta(
    label='ch4',
    basis_set='SZ',
    xc='LYP',
    mesh_cutoff=300 * Ry,
    species=[species],
    restart='ch4.XV',
    ignore_bad_restart_file=True,
    fdf_arguments={'DM.Tolerance': 1E-5,
                   'DM.MixingWeight': 0.15,
                   'DM.NumberPulay': 3,
                   'MaxSCFIterations': 200,
                   'ElectronicTemperature': 0.02585 * eV,  # 300 K
                   'SaveElectrostaticPotential': True})

bud.set_calculator(calc)
dyn = QuasiNewton(bud, trajectory=traj)
dyn.run(fmax=0.02)
e = bud.get_potential_energy()
