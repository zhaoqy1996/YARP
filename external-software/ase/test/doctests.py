import doctest
import sys

try:
    import scipy
except ImportError:
    scipy = None

from ase import atoms
from ase.collections import collection
from ase.spacegroup import spacegroup, findsym, xtal
from ase.geometry import geometry, cell
from ase.build import tools
from ase.io import ulm
import ase.eos as eos

modules = [xtal, spacegroup, cell, findsym, ulm, atoms, eos]

if scipy:
    modules.extend([geometry, tools])

if sys.version_info >= (2, 7):
    modules.append(collection)

for mod in modules:
    print(mod, doctest.testmod(mod, raise_on_error=True))
