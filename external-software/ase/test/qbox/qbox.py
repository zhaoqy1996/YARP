"""Tests related to QBOX"""

import numpy as np

from ase import Atoms
from ase.io import qbox
from ase.io import formats

# We don't like shipping raw datafiles, because they must all be listed
# in the manifest.  So we invoke a function that prepares the files that
# we need:
from ase.test.qbox.qboxdata import writefiles
writefiles()

test_qbox = 'test.xml'
test_qball = '04_md_ntc.reference.xml'


def read_output():
    """Test reading the output file"""

    # Read only one frame
    atoms = qbox.read_qbox(test_qbox)

    assert isinstance(atoms, Atoms)
    assert np.allclose(atoms.cell, np.diag([16, 16, 16]))

    assert len(atoms) == 4
    assert np.allclose(atoms[0].position,
                       [3.70001108, -0.00000000, -0.00000003],
                       atol=1e-7)  # Last frame
    assert np.allclose(atoms.get_velocities()[2],
                       [-0.00000089, -0.00000000, -0.00000000],
                       atol=1e-9)  # Last frame
    assert np.allclose(atoms.get_forces()[3],
                       [-0.00000026, -0.01699708, 0.00000746],
                       atol=1e-7)  # Last frame
    assert np.isclose(-15.37294664, atoms.get_potential_energy())
    assert np.allclose(atoms.get_stress(),
                       [-0.40353661, -1.11698386, -1.39096418,
                        0.00001786, -0.00002405, -0.00000014])

    # Read all the frames
    atoms = qbox.read_qbox(test_qbox, slice(None))

    assert isinstance(atoms, list)
    assert len(atoms) == 5

    assert len(atoms[1]) == 4
    assert np.allclose(atoms[1][0].position,
                       [3.70001108, -0.00000000, -0.00000003],
                       atol=1e-7)  # 2nd frame
    assert np.allclose(atoms[1].get_forces()[3],
                       [-0.00000029, -0.01705361, 0.00000763],
                       atol=1e-7)  # 2nd frame


def test_format():
    """Make sure the `formats.py` operations work"""

    atoms = formats.read(test_qbox)
    assert len(atoms) == 4

    atoms = formats.read(test_qbox, index=slice(None), format='qbox')
    assert len(atoms) == 5

    atoms = formats.read(test_qball)
    assert len(atoms) == 32



read_output()
test_format()
