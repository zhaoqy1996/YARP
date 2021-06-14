"""Example, in order to run you must place a pseudopotential 'Na.psf' in
the folder"""

from ase.units import Ry, eV, Ha
from ase.calculators.siesta import Siesta
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt

# Define the systems
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

# enter siesta input
siesta = Siesta(
    mesh_cutoff=150 * Ry,
    basis_set='DZP',
    pseudo_qualifier='',
    energy_shift=(10 * 10**-3) * eV,
    fdf_arguments={
        'SCFMustConverge': False,
        'COOP.Write': True,
        'WriteDenchar': True,
        'PAO.BasisType': 'split',
        'DM.Tolerance': 1e-4,
        'DM.MixingWeight': 0.01,
        'MaxSCFIterations': 300,
        'DM.NumberPulay': 4,
        'XML.Write': True})


Na8.set_calculator(siesta)
e = Na8.get_potential_energy()
tddft = siesta.pyscf_tddft_eels(label="siesta", jcutoff=7, iter_broadening=0.15/Ha,
            xc_code='LDA,PZ', tol_loc=1e-6, tol_biloc=1e-7, freq = np.arange(0.0, 5.0, 0.05))

# plot polarizability
fig = plt.figure(1)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(siesta.results["freq range"], siesta.results["eel spectra nonin"].imag)
ax2.plot(siesta.results["freq range"], siesta.results["eel spectra inter"].imag)

ax1.set_xlabel(r"$\omega$ (eV)")
ax2.set_xlabel(r"$\omega$ (eV)")

ax1.set_ylabel(r"Im($P_{xx}$) (au)")
ax2.set_ylabel(r"Im($P_{xx}$) (au)")

ax1.set_title(r"Non interacting")
ax2.set_title(r"Interacting")

fig.tight_layout()

plt.show()
