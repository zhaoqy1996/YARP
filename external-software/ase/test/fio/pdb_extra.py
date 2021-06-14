"""PDB parser

Test dealing with files that are not fully
compliant with the specification.

"""

import os
import warnings

import numpy as np

from ase import io

# Some things tested:
# Giant cell that would fail for split()
# No element field
# positions with no spaces

test_pdb = """REMARK   Not a real pdb file
CRYST1   30.00015000.00015000.000  90.00  90.00  90.00 P1
ATOM      1  C     1 X   1       1.000   8.000  12.000  0.00  0.00           C    
ATOM      1  C     1 X   1       2.000   6.000   4.000  0.00  0.00
ATOM      1  SI1 SIO     1       2.153  14.096   3.635  1.00  0.00      SIO 
ATOM      1    O   1     1       3.846   5.672   1.323  0.40 38.51            0
ATOM      1  C1'   T A   1      -2.481   5.354   0.000
ATOM      1 SIO  SIO     1     -11.713-201.677   9.060************      SIO2Si  
"""


def test_pdb_read():
    """Read information from pdb file."""
    with open('pdb_test.pdb', 'w') as pdb_file:
        pdb_file.write(test_pdb)
    expected_cell = [[30.0, 0.0, 0.0],
                     [0.0, 15000.0, 0.0],
                     [0.0, 0.0, 15000.0]]
    expected_positions = [[1.000, 8.000, 12.000],
                          [2.000, 6.000, 4.000],
                          [2.153, 14.096, 3.635],
                          [3.846, 5.672, 1.323],
                          [-2.481, 5.354, 0.000],
                          [-11.713, -201.677, 9.060]]
    expected_species = ['C', 'C', 'Si', 'O', 'C', 'Si']

    try:
        pdb_atoms = io.read('pdb_test.pdb')
        assert len(pdb_atoms) == 6
        assert np.allclose(pdb_atoms.cell, expected_cell)
        assert np.allclose(pdb_atoms.positions, expected_positions)
        assert pdb_atoms.get_chemical_symbols() == expected_species
        assert 'occupancy' not in pdb_atoms.arrays
    finally:
        os.unlink('pdb_test.pdb')


def test_pdb_read_with_arrays():
    """Read information from pdb file. Includes occupancy."""
    with open('pdb_test_2.pdb', 'w') as pdb_file:
        # only write lines with occupancy and bfactor
        pdb_file.write('\n'.join(test_pdb.splitlines()[:6]))
    expected_occupancy = [0.0, 0.0, 1.0, 0.4]
    expected_bfactor = [0.0, 0.0, 0.0, 38.51]

    try:
        pdb_atoms = io.read('pdb_test_2.pdb')
        assert len(pdb_atoms) == 4
        assert np.allclose(pdb_atoms.arrays['occupancy'], expected_occupancy)
        assert np.allclose(pdb_atoms.arrays['bfactor'], expected_bfactor)
    finally:
        os.unlink('pdb_test_2.pdb')


if __name__ in ('__main__', '__builtin__'):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Length of occupancy', UserWarning)
        test_pdb_read()
    test_pdb_read_with_arrays()
