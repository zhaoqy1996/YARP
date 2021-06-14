"""Tests of the major methods (HarmonicThermo, IdealGasThermo,
CrystalThermo) from the thermochemistry module."""

from ase import Atoms
from ase.build import fcc100, add_adsorbate
from ase.build import bulk
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from ase.phonons import Phonons
from ase.thermochemistry import (IdealGasThermo, HarmonicThermo,
                                 CrystalThermo)
from ase.calculators.emt import EMT

# Ideal gas thermo.
atoms = Atoms('N2',
              positions=[(0, 0, 0), (0, 0, 1.1)],
              calculator=EMT())
QuasiNewton(atoms).run(fmax=0.01)
energy = atoms.get_potential_energy()
vib = Vibrations(atoms, name='idealgasthermo-vib')
vib.run()
vib_energies = vib.get_energies()

thermo = IdealGasThermo(vib_energies=vib_energies, geometry='linear',
                        atoms=atoms, symmetrynumber=2, spin=0,
                        potentialenergy=energy)
thermo.get_gibbs_energy(temperature=298.15, pressure=2 * 101325.)

# Harmonic thermo.

atoms = fcc100('Cu', (2, 2, 2), vacuum=10.)
atoms.set_calculator(EMT())
add_adsorbate(atoms, 'Pt', 1.5, 'hollow')
atoms.set_constraint(FixAtoms(indices=[atom.index for atom in atoms
                                       if atom.symbol == 'Cu']))
QuasiNewton(atoms).run(fmax=0.01)
vib = Vibrations(atoms, name='harmonicthermo-vib',
                 indices=[atom.index for atom in atoms
                          if atom.symbol != 'Cu'])
vib.run()
vib.summary()
vib_energies = vib.get_energies()

thermo = HarmonicThermo(vib_energies=vib_energies,
                        potentialenergy=atoms.get_potential_energy())
thermo.get_helmholtz_energy(temperature=298.15)

# Crystal thermo.
atoms = bulk('Al', 'fcc', a=4.05)
calc = EMT()
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()

# Phonon calculator
N = 7
ph = Phonons(atoms, calc, supercell=(N, N, N), delta=0.05)
ph.run()

ph.read(acoustic=True)
phonon_energies, phonon_DOS = ph.dos(kpts=(4, 4, 4), npts=30,
                                     delta=5e-4)

thermo = CrystalThermo(phonon_energies=phonon_energies,
                       phonon_DOS=phonon_DOS,
                       potentialenergy=energy,
                       formula_units=4)
thermo.get_helmholtz_energy(temperature=298.15)
