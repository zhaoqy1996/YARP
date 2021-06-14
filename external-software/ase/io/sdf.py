"""Reads chemical data in SDF format (wraps the molfile format).

See https://en.wikipedia.org/wiki/Chemical_table_file#SDF
"""
from ase.atoms import Atoms
from ase.utils import basestring


def read_sdf(fileobj):
    if isinstance(fileobj, basestring):
        fileobj = open(fileobj)

    lines = fileobj.readlines()
    # first three lines header
    del lines[:3]

    L1 = lines.pop(0).split()
    natoms = int(L1[0])
    positions = []
    symbols = []
    for line in lines[:natoms]:
        x, y, z, symbol = line.split()[:4]
        symbols.append(symbol)
        positions.append([float(x), float(y), float(z)])
    return Atoms(symbols=symbols, positions=positions)
