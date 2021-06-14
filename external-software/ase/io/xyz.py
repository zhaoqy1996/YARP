"""The functions below are for reference only.
We use the implementation from extxyz module, which is backwards
compatible with standard XYZ format."""

from ase.atoms import Atoms
from ase.io.extxyz import read_extxyz as read_xyz, write_extxyz as write_xyz

__all__ = ['read_xyz', 'write_xyz']


def simple_read_xyz(fileobj, index):
    lines = fileobj.readlines()
    natoms = int(lines[0])
    nimages = len(lines) // (natoms + 2)
    for i in range(*index.indices(nimages)):
        symbols = []
        positions = []
        n = i * (natoms + 2) + 2
        for line in lines[n:n + natoms]:
            symbol, x, y, z = line.split()[:4]
            symbol = symbol.lower().capitalize()
            symbols.append(symbol)
            positions.append([float(x), float(y), float(z)])
        yield Atoms(symbols=symbols, positions=positions)


def simple_write_xyz(fileobj, images, comment=''):
    symbols = images[0].get_chemical_symbols()
    natoms = len(symbols)
    for atoms in images:
        fileobj.write('%d\n%s\n' % (natoms, comment))
        for s, (x, y, z) in zip(symbols, atoms.positions):
            fileobj.write('%-2s %22.15f %22.15f %22.15f\n' % (s, x, y, z))
