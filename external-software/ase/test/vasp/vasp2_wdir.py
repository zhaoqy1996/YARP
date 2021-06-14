"""
Run tests to ensure that the VASP txt and label arguments function correctly,
i.e. correctly sets the working directories and works in that directory.

This is conditional on the existence of the ASE_VASP_COMMAND, VASP_COMMAND
or VASP_SCRIPT environment variables

"""

import filecmp
import os
import shutil

from ase.test.vasp import installed2 as installed

from ase import Atoms
from ase.calculators.vasp import Vasp2 as Vasp

assert installed()

def compare_paths(path1, path2):
    assert os.path.abspath(path1) == os.path.abspath(path2)


# Test setup system, borrowed from vasp_co.py
d = 1.14
atoms = Atoms('CO', positions=[(0, 0, 0), (0, 0, d)],
              pbc=True)
atoms.center(vacuum=5.)

file1 = '_vasp_dummy_str.out'
file2 = '_vasp_dummy_io.out'
file3 = '_vasp_dummy_2.out'

testdir = '_dummy_txt_testdir'
label = os.path.join(testdir, 'vasp')

# Test
settings = dict(label=label,
                xc='PBE',
                prec='Low',
                algo='Fast',
                ismear=0,
                sigma=1.,
                istart=0,
                lwave=False,
                lcharg=False)

# Make 2 copies of the calculator object
calc = Vasp(**settings)
calc2 = Vasp(**settings)

# Check the calculator path is the expected path
compare_paths(calc.directory, testdir)

calc.set_txt(file1)
atoms.set_calculator(calc)
en1 = atoms.get_potential_energy()

# Check that the output files are in the correct directory
for fi in ['OUTCAR', 'CONTCAR', 'vasprun.xml']:
    fi = os.path.join(testdir, fi)
    assert os.path.isfile(fi)

# We open file2 in our current directory, so we don't want it to write
# in the label directory
with open(file2, 'w') as f:
    calc2.set_txt(f)
    atoms.set_calculator(calc2)
    atoms.get_potential_energy()


# Make sure the two outputfiles are identical
assert filecmp.cmp(os.path.join(calc.directory, file1), file2)

# Test restarting from working directory in test directory
label2 = os.path.join(testdir, file3)
calc2 = Vasp(restart=label,
             label=label2)

# Check the calculator path is the expected path
compare_paths(calc2.directory, testdir)

assert not calc2.calculation_required(calc2.atoms, ['energy', 'forces'])
en2 = calc2.get_potential_energy()

# Check that the restarted calculation didn't run, i.e. write to output file
assert not os.path.isfile(os.path.join(calc.directory, file3))

# Check that we loaded energy correctly
assert en1 == en2

# Clean up
shutil.rmtree(testdir)  # Remove dummy directory (non-empty)
os.remove(file2)
