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


geom = Atoms('C2',
             cell=[[0.21680326E+01, -0.12517142E+01, 0.000000000E+00],
                   [0.00000000E+00, 0.25034284E+01, 0.000000000E+00],
                   [0.00000000E+00, 0.00000000E+00, 0.50000000E+03]],
             positions=[(-0.722677550504, -1.251714234963, 0.),
                        (-1.445355101009, 0., 0.)],
             pbc=[True, True, False])

geom.set_calculator(CRYSTAL(label='graphene',
                            guess=True,
                            xc='PBE',
                            kpts=(1, 1, 1),
                            otherkeys=['scfdir', 'anderson',
                                       ['maxcycles', '500'],
                                       ['toldee', '6'],
                                       ['tolinteg', '7 7 7 7 14'],
                                       ['fmixing', '95']]))

final_energy = geom.get_potential_energy()
assert abs(final_energy + 2063.13266758) < 1.0
