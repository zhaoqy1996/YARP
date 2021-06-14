from __future__ import print_function
import os
import warnings

import numpy as np
from ase import Atoms
from ase.io import write, read, iread
from ase.io.formats import all_formats, get_ioformat
from ase.calculators.singlepoint import SinglePointCalculator

try:
    import matplotlib
except ImportError:
    matplotlib = 0

try:
    from lxml import etree
except ImportError:
    etree = 0

try:
    import Scientific
except ImportError:
    Scientific = 0

try:
    import netCDF4
except ImportError:
    netCDF4 = 0


def get_atoms():
    a = 5.0
    d = 1.9
    c = a / 2
    atoms = Atoms('AuH',
                  positions=[(0, c, c), (d, c, c)],
                  cell=(2 * d, a, a),
                  pbc=(1, 0, 0))
    extra = np.array([2.3, 4.2])
    atoms.set_array('extra', extra)
    atoms *= (2, 1, 1)

    # attach some results to the Atoms.
    # These are serialised by the extxyz writer.

    spc = SinglePointCalculator(atoms,
                                energy=-1.0,
                                stress=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                                forces=-1.0 * atoms.positions)
    atoms.set_calculator(spc)
    return atoms


def check(a, ref_atoms, format):
    assert abs(a.positions - ref_atoms.positions).max() < 1e-6, \
        (a.positions - ref_atoms.positions)
    if format in ['traj', 'cube', 'cfg', 'struct', 'gen', 'extxyz',
                  'db', 'json', 'trj']:
        assert abs(a.cell - ref_atoms.cell).max() < 1e-6
    if format in ['cfg', 'extxyz']:
        assert abs(a.get_array('extra') -
                   ref_atoms.get_array('extra')).max() < 1e-6
    if format in ['extxyz', 'traj', 'trj', 'db', 'json']:
        assert (a.pbc == ref_atoms.pbc).all()
        assert a.get_potential_energy() == ref_atoms.get_potential_energy()
        assert (a.get_stress() == ref_atoms.get_stress()).all()
        assert abs(a.get_forces() - ref_atoms.get_forces()).max() < 1e-12


testdir = 'tmp_io_testdir'
if os.path.isdir(testdir):
    import shutil
    shutil.rmtree(testdir)

os.mkdir(testdir)


def test(format):
    if format in ['abinit', 'castep-cell', 'dftb', 'eon', 'gaussian']:
        # Someone should do something ...
        return

    if format in ['v-sim', 'mustem']:
        # Standalone test used as not compatible with 1D periodicity
        return

    if format in ['mustem']:
        # Standalone test used as specific arguments are required
        return

    if format in ['dmol-arc', 'dmol-car', 'dmol-incoor']:
        # We have a standalone dmol test
        return

    if format in ['gif', 'mp4']:
        # Complex dependencies; see animate.py test
        return

    if format in ['postgresql', 'trj', 'vti', 'vtu']:
        # Let's not worry about these.
        return

    if not matplotlib and format in ['eps', 'png']:
        return

    if not etree and format == 'exciting':
        return

    if not Scientific and format == 'etsf':
        return

    if not netCDF4 and format == 'netcdftrajectory':
        return

    atoms = get_atoms()

    images = [atoms, atoms]

    io = get_ioformat(format)
    print('{0:20}{1}{2}{3}{4}'.format(format,
                                      ' R'[bool(io.read)],
                                      ' W'[bool(io.write)],
                                      '+1'[io.single],
                                      'SF'[io.acceptsfd]))
    fname1 = '{}/io-test.1.{}'.format(testdir, format)
    fname2 = '{}/io-test.2.{}'.format(testdir, format)
    if io.write:
        write(fname1, atoms, format=format)
        if not io.single:
            write(fname2, images, format=format)

        if io.read:
            for a in [read(fname1, format=format), read(fname1)]:
                check(a, atoms, format)

            if not io.single:
                if format in ['json', 'db']:
                    aa = read(fname2 + '@id=1') + read(fname2 + '@id=2')
                else:
                    aa = [read(fname2), read(fname2, 0)]
                aa += read(fname2, ':')
                for a in iread(fname2, format=format):
                    aa.append(a)
                assert len(aa) == 6, aa
                for a in aa:
                    check(a, atoms, format)

for format in sorted(all_formats):
    with warnings.catch_warnings():
        if format in ['proteindatabank', 'netcdftrajectory']:
            warnings.simplefilter('ignore', UserWarning)
        test(format)
