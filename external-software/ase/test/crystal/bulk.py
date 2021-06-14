from ase import Atoms
from ase.calculators.crystal import CRYSTAL

with open('basis', 'w') as fd:
    fd.write("""6 4
0 0 6 2.0 1.0
 3048.0 0.001826
 456.4 0.01406
 103.7 0.06876
 29.23 0.2304
 9.349 0.4685
 3.189 0.3628
0 1 2 4.0 1.0
 3.665 -0.3959 0.2365
 0.7705 1.216 0.8606
0 1 1 0.0 1.0
 0.26 1.0 1.0 
0 3 1 0.0 1.0
 0.8 1.0
""")

a0 = 5.43
bulk = Atoms('Si2', [(0, 0, 0),
                     (0.25, 0.25, 0.25)],
             pbc=True)
b = a0 / 2
bulk.set_cell([(0, b, b),
               (b, 0, b),
               (b, b, 0)], scale_atoms=True)

bulk.set_calculator(CRYSTAL(label='Si2',
                            guess=True,
                            basis='sto-3g',
                            xc='PBE',
                            kpts=(2, 2, 2),
                            otherkeys=['scfdir', 'anderson',
                                       ['maxcycles', '500'],
                                       ['toldee', '6'],
                                       ['tolinteg', '7 7 7 7 14'],
                                       ['fmixing', '50']]))

final_energy = bulk.get_potential_energy()
assert abs(final_energy + 15564.787949) < 1.0
