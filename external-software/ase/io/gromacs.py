from __future__ import print_function
""" read and write gromacs geometry files
"""

from ase.atoms import Atoms
from ase.parallel import paropen
from ase.utils import basestring
import numpy as np


def read_gromacs(filename):
    """ From:
    http://manual.gromacs.org/current/online/gro.html
    C format
    "%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f"
    python: starting from 0, including first excluding last
    0:4 5:10 10:15 15:20 20:28 28:36 36:44 44:52 52:60 60:68

    Import gromacs geometry type files (.gro).
    Reads atom positions,
    velocities(if present) and
    simulation cell (if present)
    """

    from ase.data import atomic_numbers
    from ase import units

    atoms = Atoms()
    filed = open(filename, 'r')
    lines = filed.readlines()
    filed.close()
    positions = []
    gromacs_velocities = []
    symbols = []
    tags = []
    gromacs_residuenumbers = []
    gromacs_residuenames = []
    gromacs_atomtypes = []
    sym2tag = {}
    tag = 0
    for line in (lines[2:-1]):
        #print line[0:5]+':'+line[5:11]+':'+line[11:15]+':'+line[15:20]
        # it is not a good idea to use the split method with gromacs input
        # since the fields are defined by a fixed column number. Therefore,
        # they may not be space between the fields
        #inp = line.split()

        floatvect = float(line[20:28]) * 10.0, \
            float(line[28:36]) * 10.0, \
            float(line[36:44]) * 10.0
        positions.append(floatvect)

        # read velocities
        velocities = np.array([0.0, 0.0, 0.0])
        vx = line[44:52].strip()
        vy = line[52:60].strip()
        vz = line[60:68].strip()

        for iv, vxyz in enumerate([vx, vy, vz]):
            if len(vxyz) > 0:
                try:
                    velocities[iv] = float(vxyz)
                except ValueError:
                    raise ValueError("can not convert velocity to float")
            else:
                velocities = None

        if velocities is not None:
            # velocities from nm/ps to ase units
            velocities *= units.nm / (1000.0 * units.fs)
            gromacs_velocities.append(velocities)

        gromacs_residuenumbers.append(int(line[0:5]))
        gromacs_residuenames.append(line[5:11].strip())

        symbol_read = line[11:16].strip()[0:2]
        if symbol_read not in sym2tag.keys():
            sym2tag[symbol_read] = tag
            tag += 1

        tags.append(sym2tag[symbol_read])
        if symbol_read in atomic_numbers:
            symbols.append(symbol_read)
        elif symbol_read[0] in atomic_numbers:
            symbols.append(symbol_read[0])
        elif symbol_read[-1] in atomic_numbers:
            symbols.append(symbol_read[-1])
        else:
            # not an atomic symbol
            # if we can not determine the symbol, we use
            # the dummy symbol X
            symbols.append("X")

        gromacs_atomtypes.append(line[11:16].strip())

    line = lines[-1]
    atoms = Atoms(symbols, positions, tags=tags)

    if len(gromacs_velocities) == len(atoms):
        atoms.set_velocities(gromacs_velocities)
    elif len(gromacs_velocities) != 0:
        raise ValueError("Some atoms velocities were not specified!")

    if not atoms.has('residuenumbers'):
        atoms.new_array('residuenumbers', gromacs_residuenumbers, int)
        atoms.set_array('residuenumbers',gromacs_residuenumbers, int)
    if not atoms.has('residuenames'):
        atoms.new_array('residuenames', gromacs_residuenames, str)
        atoms.set_array('residuenames', gromacs_residuenames, str)
    if not atoms.has('atomtypes'):
        atoms.new_array('atomtypes', gromacs_atomtypes, str)
        atoms.set_array('atomtypes', gromacs_atomtypes, str)


    try:
        line = lines[-1]
        inp = line.split()
        floatvect0 = \
            float(inp[0]) * 10.0, \
            float(inp[1]) * 10.0, \
            float(inp[2]) * 10.0
        try:
            floatvect1 = \
                float(inp[3]) * 10.0, \
                float(inp[4]) * 10.0, \
                float(inp[5]) * 10.0
            floatvect2 = \
                float(inp[6]) * 10.0, \
                float(inp[7]) * 10.0, \
                float(inp[8]) * 10.0
            mycell = []
            #gromacs manual (manual.gromacs.org/online/gro.html) says:
            #v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
            #
            #v1(x) v2(y) v3(z) fv0[0 1 2]  v1(x) v2(x) v3(x)
            #v1(y) v1(z) v2(x) fv1[0 1 2]  v1(y) v2(y) v3(y)
            #v2(z) v3(x) v3(y) fv2[0 1 2]  v1(z) v2(z) v3(z)
            mycell += [[floatvect0[0], floatvect1[2], floatvect2[1]]]
            mycell += [[floatvect1[0], floatvect0[1], floatvect2[2]]]
            mycell += [[floatvect1[1], floatvect2[0], floatvect0[2]]]
            atoms.set_cell(mycell)
            atoms.set_pbc(True)
        except:
            mycell = []
            #gromacs manual (manual.gromacs.org/online/gro.html) says:
            #v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
            mycell += [[floatvect0[0],           0.0,           0.0]]
            mycell += [[          0.0, floatvect0[1],           0.0]]
            mycell += [[          0.0,           0.0, floatvect0[2]]]
            atoms.set_cell(floatvect0)
            atoms.set_pbc(True)
    except:
        atoms.set_pbc(False)
    return atoms


def write_gromacs(fileobj, images):
    """Write gromacs geometry files (.gro).

    Writes:

    * atom positions,
    * velocities (if present, otherwise 0)
    * simulation cell (if present)
    """

    from ase import units

    if isinstance(fileobj, basestring):
        fileobj = paropen(fileobj, 'w')

    if not isinstance(images, (list, tuple)):
        images = [images]

    natoms = len(images[-1])
    try:
        gromacs_residuenames = images[-1].get_array('residuenames')
    except:
        gromacs_residuenames = []
        for idum in range(natoms):
            gromacs_residuenames.append('1DUM')
    try:
        gromacs_atomtypes = images[-1].get_array('atomtypes')
    except:
        gromacs_atomtypes = images[-1].get_chemical_symbols()
    try:
        residuenumbers = images[-1].get_array('residuenumbers')
    except (KeyError):
        residuenumbers = np.ones(natoms, int)

    pos = images[-1].get_positions()
    pos = pos / 10.0
    try:
        vel = images[-1].get_velocities()
        vel = vel * 1000.0 * units.fs / units.nm
    except:
        vel = pos
        vel = pos * 0.0

    # No "#" in the first line to prevent read error in VMD
    fileobj.write('A Gromacs structure file written by ASE \n')
    fileobj.write('%5d\n' % len(images[-1]))
    count = 1

    # gromac line see http://manual.gromacs.org/documentation/current/user-guide/file-formats.html#gro
    #    1WATER  OW1    1   0.126   1.624   1.679  0.1227 -0.0580  0.0434
    for resnb, resname, atomtype, xyz, vxyz in zip\
            (residuenumbers, gromacs_residuenames, gromacs_atomtypes, pos, vel):

        # THIS SHOULD BE THE CORRECT, PYTHON FORMATTING, EQUIVALENT TO THE
        # C FORMATTING GIVEN IN THE GROMACS DOCUMENTATION: 
        # >>> %5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f <<<
        line = ("{0:>5d}{1:<5s}{2:>5s}{3:>5d}{4:>8.3f}{5:>8.3f}{6:>8.3f}"
                "{7:>8.4f}{8:>8.4f}{9:>8.4f}\n"
                .format(resnb, resname, atomtype, count,
                        xyz[0], xyz[1], xyz[2], vxyz[0], vxyz[1], vxyz[2]))

        fileobj.write(line)
        count += 1

    if images[-1].get_pbc().any():
        mycell = images[-1].get_cell()
        #gromacs manual (manual.gromacs.org/online/gro.html) says:
        #v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
        #
        #cell[0,0] cell[1,0] cell[2,0] v1(x) v2(y) v3(z) fv0[0 1 2]
        #cell[0,1] cell[1,1] cell[2,1] v1(y) v1(z) v2(x) fv1[0 1 2]
        #cell[0,2] cell[1,2] cell[2,2] v2(z) v3(x) v3(y) fv2[0 1 2]
        fileobj.write('%10.5f%10.5f%10.5f' \
                          % (mycell[0, 0] * 0.1, \
                                 mycell[1, 1] * 0.1, \
                                 mycell[2, 2] * 0.1))
        fileobj.write('%10.5f%10.5f%10.5f' \
                          % (mycell[1, 0] * 0.1, \
                                 mycell[2, 0] * 0.1, \
                                 mycell[0, 1] * 0.1))
        fileobj.write('%10.5f%10.5f%10.5f\n' \
                          % (mycell[2, 1] * 0.1, \
                                 mycell[0, 2] * 0.1, \
                                 mycell[1, 2] * 0.1))
    return
