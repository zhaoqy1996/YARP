from __future__ import print_function

import numpy as np

from ase.calculators.octopus import Octopus
from ase.collections import g2
from ase.build import bulk, graphene_nanoribbon
from ase.calculators.interfacechecker import check_interface


def calculate(name, system, **kwargs):
    print('Calculate', name, system)
    label = 'ink-%s' % name

    kwargs0 = dict(stdout="'stdout.txt'",
                   FromScratch=True,
                   RestartWrite=False,
                   command='mpirun -np 4 octopus')
    kwargs.update(**kwargs0)

    calc = Octopus(label=label, **kwargs)
    system.calc = calc
    E = system.get_potential_energy()
    eig = calc.get_eigenvalues()
    check_interface(calc)

    restartcalc = Octopus(label)
    check_interface(restartcalc)

    # Check reconstruction of Atoms object
    new_atoms = restartcalc.get_atoms()
    print('new')
    print(new_atoms.positions)
    calc2 = Octopus(label='ink-restart-%s' % name, **kwargs)
    new_atoms.calc = calc2
    E2 = new_atoms.get_potential_energy()
    #print('energy', E, E2)
    eig2 = calc2.get_eigenvalues()
    eig_err = np.abs(eig - eig2).max()
    e_err = abs(E - E2)
    print('Restart E err', e_err)
    print('Restart eig err', eig_err)
    assert e_err < 5e-5
    assert eig_err < 5e-5
    return calc

if 1:
    calc = calculate('H2O',
                     g2['H2O'],
                     OutputFormat='xcrysden',
                     Output='density + wfs + potential',
                     SCFCalculateDipole=True)
    dipole = calc.get_dipole_moment()
    E = calc.get_potential_energy()

    print('dipole', dipole)
    print('energy', E)

    dipole_err = np.abs(dipole - [0., 0., -0.37]).max()
    assert dipole_err < 0.02, dipole_err
    energy_err = abs(-463.5944954 - E)
    assert energy_err < 0.001, energy_err

if 1:
    atoms = g2['O2']
    atoms.center(vacuum=2.0)
    calc = calculate('O2',
                     atoms,
                     BoxShape='parallelepiped',
                     SpinComponents='spin_polarized',
                     ExtraStates=2)
    #magmom = calc.get_magnetic_moment()
    #magmoms = calc.get_magnetic_moments()
    #print('magmom', magmom)
    #print('magmoms', magmoms)
if 1:
    calc = calculate('Si',
                     bulk('Si', orthorhombic=True),
                     KPointsGrid=[[4, 4, 4]],
                     KPointsUseSymmetries=True,
                     SmearingFunction='fermi_dirac',
                     ExtraStates=2,
                     Smearing='0.1 * eV',
                     ExperimentalFeatures=True,
                     Spacing='0.35 * Angstrom')
    eF = calc.get_fermi_level()
    print('eF', eF)
if 0:  # This calculation does not run will in Octopus
    # We will do the "toothless" spin-polarised Si instead.
    calc = calculate('Fe',
                     bulk('Fe', orthorhombic=True),
                     KPointsGrid=[[4, 4, 4]],
                     KPointsUseSymmetries=True,
                     ExtraStates=4,
                     Spacing='0.15 * Angstrom',
                     SmearingFunction='fermi_dirac',
                     Smearing='0.1 * eV',
                     PseudoPotentialSet='sg15',
                     ExperimentalFeatures=True,
                     SpinComponents='spin_polarized')
    eF = calc.get_fermi_level()
    assert abs(eF - 5.33) < 1e-1
    # XXXX octopus does not get magnetic state?
if 1:
    calc = calculate('Si',
                     bulk('Si', orthorhombic=True),
                     KPointsGrid=[[4, 4, 4]],
                     SpinComponents='spin_polarized',
                     ExtraStates=2,
                     SmearingFunction='fermi_dirac',
                     Smearing='0.1 * eV',
                     KPointsUseSymmetries=True,
                     ExperimentalFeatures=True,
                     Spacing='0.35 * Angstrom')
    #eF = calc.get_fermi_level()
    print('eF', eF)

if 0:
    # Experimental feature: mixed periodicity.  Let us not do this for now...
    graphene = graphene_nanoribbon(2, 2, sheet=True)
    graphene.positions = graphene.positions[:, [0, 2, 1]]
    graphene.pbc = [1, 1, 0] # from 1, 0, 1
    calc = calculate('graphene',
                     graphene,
                     KPointsGrid=[[2, 1, 2]],
                     KPointsUseSymmetries=True,
                     ExperimentalFeatures=True,
                     ExtraStates=4,
                     SmearingFunction='fermi_dirac',
                     Smearing='0.1 * eV')
