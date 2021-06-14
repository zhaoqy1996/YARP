from ase import Atoms
from ase.io import read, write
from ase.calculators.exciting import Exciting


a = Atoms('N3O',
          [(0, 0, 0), (1, 0, 0), (0, 0, 1), (0.5, 0.5, 0.5)],
          pbc=True)

write('geo.exi', a)
b = read('geo.exi')

print(a)
print(a.get_positions())
print(b)
print(b.get_positions())

calculator = Exciting(dir='excitingtestfiles',
                      kpts=(4, 4, 3),
                      # bin='/fshome/chm/git/exciting/bin/excitingser',
                      maxscl=3)
