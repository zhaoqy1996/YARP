# This writes xsd example with bond connectivity information, and checks
# bond formats.


from ase import Atoms
from ase.io import write
import numpy as np
from collections import OrderedDict
import re
# Example molecule
atoms = Atoms('CH4',[[ 1.08288111e-09, 1.74602682e-09,-1.54703448e-09],
    [-6.78446715e-01, 8.73516584e-01,-8.63073811e-02],
    [-4.09602527e-01,-8.46016530e-01,-5.89280858e-01],
    [ 8.52016070e-02,-2.98243876e-01, 1.06515792e+00],
    [ 1.00284763e+00, 2.70743821e-01,-3.89569679e-01]])
connectivitymatrix = np.array([[0, 1, 1, 1, 1], # Carbon(index 0), is connected to other hydrogen atoms (index 1-4)
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0], 
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0]])
write('xsd_test_CH4.xsd',atoms,connectivity = connectivitymatrix)

# Read and see if the atom information and bond information matches.
AtomIdsToBondIds = OrderedDict()
BondIdsToConnectedAtomIds = OrderedDict()
with open('xsd_test_CH4.xsd','r') as f:
    for i,line in enumerate(f):
        if '<Atom3d ' in line:
            AtomId = int(re.search(r'ID="(.*?)"', line).group(1))
            ConnectedBondIds = [int(a) for a in re.search(r'Connections="(.*?)"', line).group(1).split(',')]
            AtomIdsToBondIds[AtomId] = ConnectedBondIds
        elif '<Bond ' in line:
            BondId = int(re.search(r'ID="(.*?)"', line).group(1))
            ConnectedAtomIds = [int(a) for a in re.search(r'Connects="(.*?)"', line).group(1).split(',')]
            BondIdsToConnectedAtomIds[BondId] = ConnectedAtomIds
# check if atom ids have been correctly assigned for each bond
for AtomId in AtomIdsToBondIds:
    for BondId in AtomIdsToBondIds[AtomId]:
        assert AtomId in BondIdsToConnectedAtomIds[BondId]

# make connectivity graph and see if it matches with input.
AtomIds = list(AtomIdsToBondIds.keys())
Newconnectivitymatrix = np.zeros((5,5))
for AtomId in AtomIdsToBondIds:
    for BondId in AtomIdsToBondIds[AtomId]:
        OtherAtomId = [a  for a in BondIdsToConnectedAtomIds[BondId] if a != AtomId]
        i = AtomIds.index(AtomId)
        j = AtomIds.index(OtherAtomId[0])
        Newconnectivitymatrix[i,j] = 1
for i in range(0,4):
    for j in range(0,4):
        assert connectivitymatrix[i,j] == Newconnectivitymatrix[i,j]


