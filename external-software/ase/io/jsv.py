"""
A module for reading and writing crystal structures from JSV
See http://www.jcrystal.com/steffenweber/JAVA/JSV/jsv.html

By Jesper Friis, Jan. 2012
"""


import re

import numpy as np

import ase
from ase.spacegroup import Spacegroup, crystal
from ase.geometry import cellpar_to_cell, cell_to_cellpar


def read_jsv(f):
    """Reads a JSV file."""
    natom = nbond = npoly = 0
    symbols = []
    labels = []
    cellpar = basis = title = bonds = poly = origin = shell_numbers = None
    spacegroup = 1

    headline = f.readline().strip()

    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()
        m = re.match(r'^\[([^]]+)\]\s*(.*)', line)
        if m is None or not line:
            continue
        tag = m.groups()[0].lower()

        if len(m.groups()) > 1:
            args = m.groups()[1].split()
        else:
            args = []

        if tag == 'cell':
            cellpar = [float(x) for x in args]
        elif tag == 'natom':
            natom = int(args[0])
        elif tag == 'nbond':
            nbond = int(args[0])
            # optional margin of the bondlengths
        elif tag == 'npoly':
            npoly = int(args[0])
        elif tag == 'space_group':
            spacegroup = Spacegroup(*tuple(int(x) for x in args))
        elif tag == 'title':
            title = m.groups()[1]
        elif tag == 'atoms':
            symbols = []
            basis = np.zeros((natom, 3), dtype=float)
            shell_numbers = -np.ones((natom, ), dtype=int)  # float?
            for i in range(natom):
                tokens = f.readline().strip().split()
                labels.append(tokens[0])
                symbols.append(ase.data.chemical_symbols[int(tokens[1])])
                basis[i] = [float(x) for x in tokens[2:5]]
                if len(tokens) > 5:
                    shell_numbers[i] = float(tokens[5])  # float?
        elif tag == 'bonds':
            for i in range(nbond):
                f.readline()
            bonds = NotImplemented
        elif tag == 'poly':
            for i in range(npoly):
                f.readline()
            poly = NotImplemented
        elif tag == 'origin':
            origin = NotImplemented
        else:
            raise ValueError('Unknown tag: "%s"' % tag)

    if headline == 'asymmetric_unit_cell':
        atoms = crystal(symbols=symbols,
                        basis=basis,
                        spacegroup=spacegroup,
                        cellpar=cellpar,
                        )
    elif headline == 'full_unit_cell':
        atoms = ase.Atoms(symbols=symbols,
                          scaled_positions=basis,
                          cell=cellpar_to_cell(cellpar),
                          )
        atoms.info['spacegroup'] = Spacegroup(spacegroup)
    elif headline == 'cartesian_cell':
        atoms = ase.Atoms(symbols=symbols,
                          positions=basis,
                          cell=cellpar_to_cell(cellpar),
                          )
        atoms.info['spacegroup'] = Spacegroup(spacegroup)
    else:
        raise ValueError('Invalid JSV file type: "%s"' % headline)

    atoms.info['title'] = title
    atoms.info['labels'] = labels
    if bonds is not None:
        atoms.info['bonds'] = bonds
    if poly is not None:
        atoms.info['poly'] = poly
    if origin is not None:
        atoms.info['origin'] = origin
    if shell_numbers is not None:
        atoms.info['shell_numbers'] = shell_numbers

    return atoms


def write_jsv(f, atoms):
    """Writes JSV file."""
    f.write('asymmetric_unit_cell\n')

    f.write('[cell]')
    for v in cell_to_cellpar(atoms.cell):
        f.write('  %g' % v)
    f.write('\n')

    f.write('[natom]  %d\n' % len(atoms))
    f.write('[nbond]  0\n')  # FIXME
    f.write('[npoly]  0\n')  # FIXME

    if 'spacegroup' in atoms.info:
        sg = Spacegroup(atoms.info['spacegroup'])
        f.write('[space_group]  %d %d\n' % (sg.no, sg.setting))
    else:
        f.write('[space_group]  1  1\n')

    f.write('[title] %s\n' % atoms.info.get('title', 'untitled'))

    f.write('\n')
    f.write('[atoms]\n')
    if 'labels' in atoms.info:
        labels = atoms.info['labels']
    else:
        labels = ['%s%d' % (s, i + 1) for i, s in
                  enumerate(atoms.get_chemical_symbols())]
    numbers = atoms.get_atomic_numbers()
    scaled = atoms.get_scaled_positions()
    for l, n, p in zip(labels, numbers, scaled):
        f.write('%-4s  %2d  %9.6f  %9.6f  %9.6f\n' % (l, n, p[0], p[1], p[2]))

    f.write('Label  AtomicNumber  x y z (repeat natom times)\n')

    f.write('\n')
    f.write('[bonds]\n')

    f.write('\n')
    f.write('[poly]\n')

    f.write('\n')
