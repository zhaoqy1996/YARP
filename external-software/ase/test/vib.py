from __future__ import print_function
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo

n2 = Atoms('N2',
           positions=[(0, 0, 0), (0, 0, 1.1)],
           calculator=EMT())
QuasiNewton(n2).run(fmax=0.01)
vib = Vibrations(n2)
vib.run()
print(vib.get_frequencies())
vib.summary()
print(vib.get_mode(-1))
vib.write_mode(n=None, nimages=20)
vib_energies = vib.get_energies()

for image in vib.iterimages():
    assert len(image) == 2

thermo = IdealGasThermo(vib_energies=vib_energies, geometry='linear',
                        atoms=n2, symmetrynumber=2, spin=0)
thermo.get_gibbs_energy(temperature=298.15, pressure=2 * 101325.)

assert vib.clean(empty_files=True) == 0
assert vib.clean() == 13
assert len(list(vib.iterimages())) == 13

d = dict(vib.iterdisplace(inplace=False))

for name, atoms in vib.iterdisplace(inplace=True):
    assert d[name] == atoms
