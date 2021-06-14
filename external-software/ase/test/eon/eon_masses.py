"""Check that reading and writing masses in .con files is consistent."""

import tempfile
import os
import shutil

from numpy import asarray
import ase.lattice.compounds
import ase.data
import ase.io


# Error tolerance.
TOL = 1e-8

data = ase.lattice.compounds.B2(['Cs', 'Cl'], latticeconstant=4.123,
                                size=(3, 3, 3))

m_Cs = ase.data.atomic_masses[ase.data.atomic_numbers['Cs']]
m_Cl = ase.data.atomic_masses[ase.data.atomic_numbers['Cl']]


tempdir = tempfile.mkdtemp()
try:
    con_file = os.path.join(tempdir, 'pos.con')
    # Write and read the .con file.
    ase.io.write(con_file, data, format='eon')
    data2 = ase.io.read(con_file, format='eon')
    # Check masses.
    symbols = asarray(data2.get_chemical_symbols())
    masses = asarray(data2.get_masses())
    assert (abs(masses[symbols == 'Cs'] - m_Cs)).sum() < TOL
    assert (abs(masses[symbols == 'Cl'] - m_Cl)).sum() < TOL
finally:
    shutil.rmtree(tempdir)
