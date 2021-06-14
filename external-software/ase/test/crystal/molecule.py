from ase.optimize import BFGS
from ase.atoms import Atoms
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

geom = Atoms('OHH',
             positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)])

geom.set_calculator(CRYSTAL(label='water',
                            guess=True,
                            basis='sto-3g',
                            xc='PBE',
                            otherkeys=['scfdir', 'anderson',
                                       ['maxcycles', '500'],
                                       ['toldee', '6'],
                                       ['tolinteg', '7 7 7 7 14'],
                                       ['fmixing', '90']]))

opt = BFGS(geom)
opt.run(fmax=0.05)

final_energy = geom.get_potential_energy()
assert abs(final_energy + 2047.34531091) < 1.0
