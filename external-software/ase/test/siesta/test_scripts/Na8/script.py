"""Example, in order to run you must place a pseudopotential 'Na.psf' in
the folder"""

from ase.units import Ry, eV
from ase.calculators.siesta import Siesta
from ase import Atoms

Na8 = Atoms('Na8',
            positions=[[-1.90503810, 1.56107288, 0.00000000],
                       [1.90503810, 1.56107288, 0.00000000],
                       [1.90503810, -1.56107288, 0.00000000],
                       [-1.90503810, -1.56107288, 0.00000000],
                       [0.00000000, 0.00000000, 2.08495836],
                       [0.00000000, 0.00000000, -2.08495836],
                       [0.00000000, 3.22798122, 2.08495836],
                       [0.00000000, 3.22798122, -2.08495836]],
            cell=[20, 20, 20])

siesta = Siesta(
    mesh_cutoff=150 * Ry,
    basis_set='DZP',
    energy_shift=(10 * 10**-3) * eV,
    fdf_arguments={
        'SCFMustConverge': False,
        'COOP.Write': True,
        'WriteDenchar': True,
        'PAO.BasisType': 'split',
        'DM.Tolerance': 1e-4,
        'DM.MixingWeight': 0.01,
        'MaxSCFIterations': 3,
        'DM.NumberPulay': 4})

Na8.set_calculator(siesta)
print(Na8.get_potential_energy())

print(siesta.results['fermi_energy'])
print(siesta.results['dim'].natoms_interacting)
print(siesta.results['pld'].cell)
print(siesta.results['wfsx'].norbitals)

for key in siesta.results['ion'].keys():
    print(key, siesta.results['ion'][key].keys())
