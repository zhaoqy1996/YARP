from ase import Atoms
from ase.units import Ry
from ase.calculators.abinit import Abinit

a0 = 5.43
bulk = Atoms('Si2', [(0, 0, 0),
                     (0.25, 0.25, 0.25)],
             pbc=True)
b = a0 / 2
bulk.set_cell([(0, b, b),
               (b, 0, b),
               (b, b, 0)], scale_atoms=True)

calc = Abinit(label='Si',
              nbands=8,  # one can specify any abinit keywords
              ecut=10 * Ry,  # warning - used to speedup the test
              kpts=[4, 4, 4],  # warning - used to speedup the test
              chksymbreak=0,
              )

# one can specify abinit keywords also using set
calc.set(toldfe=1.0e-2)  # warning - used to speedup the test
bulk.set_calculator(calc)
e = bulk.get_potential_energy()
