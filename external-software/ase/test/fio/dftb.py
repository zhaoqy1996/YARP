# additional tests of the dftb I/O
import numpy as np
from ase.io.dftb import read_dftb_lattice
from ase.atoms import Atoms
from io import StringIO

#test ase.io.dftb.read_dftb_lattice
fd = StringIO(u"""
 MD step: 0
 Lattice vectors (A)
  26.1849388999576 5.773808884828536E-006 9.076696618724854E-006  
 0.115834159141441 26.1947703089401 9.372892011565608E-006
 0.635711495837792 0.451552307731081 9.42069476334197
 Volume:                             0.436056E+05 au^3   0.646168E+04 A^3
 Pressure:                           0.523540E-04 au     0.154031E+10 Pa
 Gibbs free energy:               -374.4577147047 H       -10189.5129 eV
 Gibbs free energy including KE   -374.0819244147 H       -10179.2871 eV
 Potential Energy:                -374.4578629171 H       -10189.5169 eV
 MD Kinetic Energy:                  0.3757902900 H           10.2258 eV
 Total MD Energy:                 -374.0820726271 H       -10179.2911 eV
 MD Temperature:                     0.0009525736 au         300.7986 K
 MD step: 10
 Lattice vectors (A)
 26.1852379966047 5.130835479368833E-005 5.227350674663197E-005
 0.115884270570380 26.1953147133737 7.278784404810537E-005
 0.635711495837792 0.451552307731081 9.42069476334197
 Volume:                             0.436085E+05 au^3   0.646211E+04 A^3
 Pressure:                           0.281638E-04 au     0.828608E+09 Pa
 Gibbs free energy:               -374.5467030749 H       -10191.9344 eV
 Gibbs free energy including KE   -374.1009478784 H       -10179.8047 eV
 Potential Energy:                -374.5468512972 H       -10191.9384 eV
 MD Kinetic Energy:                  0.4457551965 H           12.1296 eV
 Total MD Energy:                 -374.1010961007 H       -10179.8088 eV
 MD Temperature:                     0.0011299245 au         356.8015 K
""")

vectors = read_dftb_lattice(fd)
mols = [Atoms(),Atoms()]
read_dftb_lattice(fd,mols)

compareVec = np.array([[26.1849388999576,5.773808884828536E-006,9.076696618724854E-006],[0.115834159141441,26.1947703089401,9.372892011565608E-006],[0.635711495837792,0.451552307731081,9.42069476334197]])

assert (vectors[0] == compareVec).all()
assert len(vectors) == 2
assert len(vectors[1]) == 3
assert (mols[0].get_cell() == compareVec).all()
assert mols[1].get_pbc().all() == True
