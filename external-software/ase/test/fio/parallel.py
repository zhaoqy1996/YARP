from ase import Atoms
from ase.io import read, write
from ase.parallel import world

n = world.rank + 1
a = Atoms('H' * n)
name = 'H{}.xyz'.format(n)
write(name, a, parallel=False)
b = read(name, parallel=False)
assert n == len(b)
