"""Example, in order to run you must place a pseudopotential 'Na.psf' in
the folder"""

from ase.units import Ry, eV
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
        'DM.NumberPulay': 4})


mbpt_inp = {'prod_basis_type': 'MIXED',
            'solver_type': 1,
            'gmres_eps': 0.001,
            'gmres_itermax': 256,
            'gmres_restart': 250,
            'gmres_verbose': 20,
            'xc_ord_lebedev': 14,
            'xc_ord_gl': 48,
            'nr': 512,
            'akmx': 100,
            'eigmin_local': 1e-06,
            'eigmin_bilocal': 1e-08,
            'freq_eps_win1': 0.15,
            'd_omega_win1': 0.05,
            'dt': 0.1,
            'omega_max_win1': 5.0,
            'ext_field_direction': 2,
            'dr': np.array([0.3, 0.3, 0.3]),
            'para_type': 'MATRIX',
            'chi0_v_algorithm': 14,
            'format_output': 'text',
            'comp_dens_chng_and_polarizability': 1,
            'store_dens_chng': 1,
            'enh_given_volume_and_freq': 0,
            'diag_hs': 0,
            'do_tddft_tem': 0,
            'do_tddft_iter': 1,
            'plot_freq': 3.02,
            'gwa_initialization': 'SIESTA_PB'}


Na8.set_calculator(siesta)
e = Na8.get_potential_energy()
freq, pol = siesta.get_polarizability(mbpt_inp,
                                      format_output='txt',
                                      units='nm**2')

# plot polarizability
plt.plot(freq, pol[:, 0, 0].im)

plt.show()
