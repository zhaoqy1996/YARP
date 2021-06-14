"""Test that amber calculator works.

This is conditional on the existence of the $AMBERHOME/bin/sander
executable.
"""
import subprocess

from ase import Atoms
from ase.calculators.amber import Amber
from ase.test import require


require('amber')

with open('mm.in', 'w') as outfile:
    outfile.write("""\
zero step md to get energy and force
&cntrl
imin=0, nstlim=0,  ntx=1 !0 step md
cut=100, ntb=0,          !non-periodic
ntpr=1,ntwf=1,ntwe=1,ntwx=1 ! (output frequencies)
&end
END
""")

with open('tleap.in', 'w') as outfile:
    outfile.write("""\
source leaprc.protein.ff14SB
source leaprc.gaff
source leaprc.water.tip3p
mol = loadpdb 2h2o.pdb
saveamberparm mol 2h2o.top h2o.inpcrd
quit
""")

subprocess.call('tleap -f tleap.in'.split())

atoms = Atoms('OH2OH2',
              [[-0.956, -0.121, 0],
               [-1.308, 0.770, 0],
               [0.000, 0.000, 0],
               [3.903, 0.000, 0],
               [4.215, -0.497, -0.759],
               [4.215, -0.497, 0.759]])

calc = Amber(amber_exe='sander -O ',
             infile='mm.in',
             outfile='mm.out',
             topologyfile='2h2o.top',
             incoordfile='mm.crd')
calc.write_coordinates(atoms, 'mm.crd')
atoms.set_calculator(calc)

e = atoms.get_potential_energy()
assert abs(e + 0.046799672) < 5e-3
