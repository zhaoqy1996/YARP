""" test run for gromacs calculator """

from ase.test import NotAvailable
from ase.calculators.gromacs import Gromacs

if Gromacs().get_command() is None:
    raise NotAvailable(
        'Gromacs required, setup your GMXCMD environmental variable')

GRO_INIT_FILE = 'hise_box.gro'

# write structure file
outfile = open('hise_box.gro', 'w')
outfile.write('HISE for testing  \n')
outfile.write('   20 \n')
outfile.write('    3HISE     N    1   1.966   1.938   1.722 \n')
outfile.write('    3HISE    H1    2   2.053   1.892   1.711 \n')
outfile.write('    3HISE    H2    3   1.893   1.882   1.683 \n')
outfile.write('    3HISE    H3    4   1.969   2.026   1.675 \n')
outfile.write('    3HISE    CA    5   1.939   1.960   1.866 \n')
outfile.write('    3HISE    HA    6   1.934   1.869   1.907 \n')
outfile.write('    3HISE    CB    7   2.055   2.041   1.927 \n')
outfile.write('    3HISE   HB1    8   2.141   2.007   1.890 \n')
outfile.write('    3HISE   HB2    9   2.043   2.137   1.903 \n')
outfile.write('    3HISE   ND1   10   1.962   2.069   2.161 \n')
outfile.write('    3HISE    CG   11   2.065   2.032   2.077 \n')
outfile.write('    3HISE   CE1   12   2.000   2.050   2.287 \n')
outfile.write('    3HISE   HE1   13   1.944   2.069   2.368 \n')
outfile.write('    3HISE   NE2   14   2.123   2.004   2.287 \n')
outfile.write('    3HISE   HE2   15   2.177   1.981   2.369 \n')
outfile.write('    3HISE   CD2   16   2.166   1.991   2.157 \n')
outfile.write('    3HISE   HD2   17   2.256   1.958   2.128 \n')
outfile.write('    3HISE     C   18   1.806   2.032   1.888 \n')
outfile.write('    3HISE   OT1   19   1.736   2.000   1.987 \n')
outfile.write('    3HISE   OT2   20   1.770   2.057   2.016 \n')
outfile.write('   4.00000   4.00000   4.00000 \n')
outfile.close()


CALC_MM_RELAX = Gromacs(
    force_field='charmm27',
    define='-DFLEXIBLE',
    integrator='cg',
    nsteps='10000',
    nstfout='10',
    nstlog='10',
    nstenergy='10',
    nstlist='10',
    ns_type='grid',
    pbc='xyz',
    rlist='0.7',
    coulombtype='PME-Switch',
    rcoulomb='0.6',
    vdwtype='shift',
    rvdw='0.6',
    rvdw_switch='0.55',
    DispCorr='Ener')
CALC_MM_RELAX.set_own_params_runs(
    'init_structure', 'hise_box.gro')
CALC_MM_RELAX.generate_topology_and_g96file()
CALC_MM_RELAX.write_input()
CALC_MM_RELAX.generate_gromacs_run_file()
CALC_MM_RELAX.run()
atoms = CALC_MM_RELAX.get_atoms()
final_energy = CALC_MM_RELAX.get_potential_energy(atoms)

assert abs(final_energy + 4.06503308131) < 5e-3
