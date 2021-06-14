from __future__ import print_function
from ase import Atoms
from ase.calculators.siesta import Siesta
from ase.units import Ry

a0 = 5.43
bulk = Atoms('Si2', [(0, 0, 0),
                     (0.25, 0.25, 0.25)],
             pbc=True)
b = a0 / 2
bulk.set_cell([(0, b, b),
               (b, 0, b),
               (b, b, 0)], scale_atoms=True)

calc = Siesta(label='Si',
              xc='PBE',
              mesh_cutoff=200 * Ry,
              energy_shift=0.01 * Ry,
              basis_set='DZ',
              kpts=[1, 2, 3],
              fdf_arguments={'DM.MixingWeight': 0.10,
                             'MaxSCFIterations': 10,
                             'DM.Tolerance': 0.1,
                             },
              )
bulk.set_calculator(calc)
e = bulk.get_potential_energy()
print(e)
