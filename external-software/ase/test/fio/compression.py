"""
Read and write on compressed files.
"""

import os
import os.path

import numpy as np

from ase import io
from ase.io import formats
from ase.build import bulk
from ase.test import NotAvailable


single = bulk('Au')
multiple = [bulk('Fe'), bulk('Zn'), bulk('Li')]


def test_get_compression():
    """Identification of supported compression from filename."""
    assert formats.get_compression('H2O.pdb.gz') == ('H2O.pdb', 'gz')
    assert formats.get_compression('CH4.pdb.bz2') == ('CH4.pdb', 'bz2')
    assert formats.get_compression('Alanine.pdb.xz') == ('Alanine.pdb', 'xz')
    # zip not implemented ;)
    assert formats.get_compression('DNA.pdb.zip') == ('DNA.pdb.zip', None)
    assert formats.get_compression('crystal.cif') == ('crystal.cif', None)


def test_compression_write_single(ext='gz'):
    """Writing compressed file."""
    filename = 'single.xsf.{ext}'.format(ext=ext)
    io.write(filename, single)
    assert os.path.exists(filename)
    os.unlink(filename)


def test_compression_read_write_single(ext='gz'):
    """Re-reading a compressed file."""
    # Use xsf filetype as it needs to check the 'magic'
    # filetype guessing when reading
    filename = 'single.xsf.{ext}'.format(ext=ext)
    io.write(filename, single)
    assert os.path.exists(filename)
    reread = io.read(filename)
    assert reread.get_chemical_symbols() == single.get_chemical_symbols()
    assert np.allclose(reread.positions, single.positions)
    os.unlink(filename)


def test_compression_write_multiple(ext='gz'):
    """Writing compressed file, with multiple configurations."""
    filename = 'multiple.xyz.{ext}'.format(ext=ext)
    io.write(filename, multiple)
    assert os.path.exists(filename)
    os.unlink(filename)


def test_compression_read_write_multiple(ext='gz'):
    """Re-reading a compressed file with multiple configurations."""
    filename = 'multiple.xyz.{ext}'.format(ext=ext)
    io.write(filename, multiple)
    assert os.path.exists(filename)
    reread = io.read(filename, ':')
    assert len(reread) == len(multiple)
    assert np.allclose(reread[-1].positions, multiple[-1].positions)
    os.unlink(filename)


def test_modes(ext='gz'):
    """Test the different read/write modes for a compression format."""
    filename = 'testrw.{ext}'.format(ext=ext)
    for mode in ['w', 'wb', 'wt']:
        with formats.open_with_compression(filename, mode) as tmp:
            if 'b' in mode:
                tmp.write(b'some text')
            else:
                tmp.write('some text')

    for mode in ['r', 'rb', 'rt']:
        with formats.open_with_compression(filename, mode) as tmp:
            if 'b' in mode:
                assert tmp.read() == b'some text'
            else:
                assert tmp.read() == 'some text'

    os.unlink(filename)


if __name__ in ('__main__', '__builtin__'):
    test_get_compression()
    # gzip
    test_compression_write_single()
    test_compression_read_write_single()
    test_compression_write_multiple()
    test_compression_read_write_multiple()
    test_modes()
    # bz2
    test_compression_write_single('bz2')
    test_compression_read_write_single('bz2')
    test_compression_write_multiple('bz2')
    test_compression_read_write_multiple('bz2')
    test_modes('bz2')
    # xz
    # These will fail in Python 2 if backports.lzma is not installed,
    # but raise different errors depending on whether any other
    # backports modules are installed. Catch here so the skip message
    # always has both parts of the module name.
    # Do xz last so the other formats are always tested anyway.
    try:
        test_compression_write_single('xz')
        test_compression_read_write_single('xz')
        test_compression_write_multiple('xz')
        test_compression_read_write_multiple('xz')
        test_modes('xz')
    except ImportError as ex:
        if 'lzma' in ex.args[0] or 'backports' in ex.args[0]:
            raise NotAvailable('no backports.lzma module')
        else:
            raise
