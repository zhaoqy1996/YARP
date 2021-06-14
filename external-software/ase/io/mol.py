"""Reads chemical data in MDL Molfile format. 

See https://en.wikipedia.org/wiki/Chemical_table_file
"""
from ase.atoms import Atoms


def read_mol(fileobj):
    lines = fileobj.readlines()
    del(lines[:3])
    L1 = lines[0].split()
    del(lines[0])
    natoms = int(L1[0])
    positions = []
    symbols = []
    for line in lines[:natoms]:
            x, y, z, symbol = line.split()[:4]
            symbols.append(symbol)
            positions.append([float(x), float(y), float(z)])
    return Atoms(symbols=symbols, positions=positions)
