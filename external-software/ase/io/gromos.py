from __future__ import print_function
""" write gromos96 geometry files
(the exact file format is copied from the freely available
gromacs package, http://www.gromacs.org
its procedure src/gmxlib/confio.c (write_g96_conf)
"""

from ase.parallel import paropen
from ase.utils import basestring


def read_gromos(fileobj, index=-1):
    """Read gromos geometry files (.g96).
    Reads:
    atom positions,
    and simulation cell (if present)
    tries to set atom types
    """

    from ase import Atoms
    from ase.data import chemical_symbols
    import sys

    if isinstance(fileobj, basestring):
        fileobj = paropen(fileobj, 'r')

    if (index != -1):
        print('In gromos (g96) format only last frame can be read')
        sys.exit()

    lines = fileobj.readlines()
    read_pos = False
    read_box = False
    tmp_pos = []
    symbols = []
    mycell = None
    for line in lines:
        if (read_pos and ('END' in line)):
            read_pos = False
        if (read_box and ('END' in line)):
            read_box = False
        if read_pos:
            symbol, dummy, x, y, z = line.split()[2:7]
            tmp_pos.append((10*float(x), 10*float(y), 10*float(z)))
            if (len(symbol) != 2):
                symbols.append(symbol[0].lower().capitalize())
            else:
                symbol2 = symbol[0].lower().capitalize() + \
                    symbol[1]
                if symbol2 in chemical_symbols:
                    symbols.append(symbol2)
                else:
                    symbols.append(symbol[0].lower().capitalize())
            if symbols[-1] not in chemical_symbols:
                print('Symbol not in chemical symbols, please check',\
                    symbols[-1])
                sys.exit()
        if read_box:
            try:
                b00, b11, b22, b10, b20, b01, b21, b02, b12 = line.split()[:9]
                mycell = [(10.*float(b00), 10.*float(b01), 10.*float(b02)),
                          (10.*float(b10), 10.*float(b11), 10.*float(b12)),
                          (10.*float(b20), 10.*float(b21), 10.*float(b22))]
            except:
                b00, b11, b22 = line.split()[:3]
                mycell = [(10.*float(b00), 0.0, 0.0),
                          (0.0, 10.*float(b11), 0.0),
                          (0.0, 0.0, 10.*float(b22))]
        if ('POSITION' in line):
            read_pos = True
        if ('BOX' in line):
            read_box = True
    if mycell == None:
        gmx_system = Atoms(symbols=symbols, positions=tmp_pos)
    else:
        gmx_system = Atoms(symbols=symbols, positions=tmp_pos, cell=mycell)
        gmx_system.set_pbc(True)
    return gmx_system

def write_gromos(fileobj, images):
    """Write gromos geometry files (.g96).
    Writes:
    atom positions,
    and simulation cell (if present)
    """

    from ase import units

    if isinstance(fileobj, basestring):
        fileobj = paropen(fileobj, 'w')

    if not isinstance(images, (list, tuple)):
        images = [images]

    natoms = len(images[-1])
    try:
        gromos_residuenames = images[-1].get_array('residuenames')
    except:
        gromos_residuenames = []
        for idum in range(natoms):
            gromos_residuenames.append('1DUM')
    try:
        gromos_atomtypes = images[-1].get_array('atomtypes')
    except:
        gromos_atomtypes = images[-1].get_chemical_symbols()

    pos = images[-1].get_positions()
    pos = pos / 10.0
    try:
        vel = images[-1].get_velocities()
        vel = vel * 1000.0 * units.fs / units.nm
    except:
        vel = pos
        vel = pos * 0.0

    fileobj.write('TITLE\n')
    fileobj.write('Gromos96 structure file written by ASE \n')
    fileobj.write('END\n')
    fileobj.write('POSITION\n')
    count = 1
    rescount = 0
    oldresname = ''
    for resname, atomtype, xyz in zip\
            (gromos_residuenames, gromos_atomtypes, pos):
        if resname != oldresname:
            oldresname = resname
            rescount = rescount + 1
        okresname = resname.lstrip('0123456789 ')
        fileobj.write('%5d %-5s %-5s%7d%15.9f%15.9f%15.9f\n' % \
                          (rescount, okresname, atomtype, count, \
                               xyz[0], xyz[1], xyz[2]))
        count = count + 1
    fileobj.write('END\n')

    if images[-1].get_pbc().any():
        fileobj.write('BOX\n')
        mycell = images[-1].get_cell()
        fileobj.write('%15.9f%15.9f%15.9f' \
                          % (mycell[0, 0] * 0.1, \
                                 mycell[1, 1] * 0.1, \
                                 mycell[2, 2] * 0.1))
        fileobj.write('%15.9f%15.9f%15.9f' \
                          % (mycell[1, 0] * 0.1, \
                                 mycell[2, 0] * 0.1, \
                                 mycell[0, 1] * 0.1))
        fileobj.write('%15.9f%15.9f%15.9f\n' \
                          % (mycell[2, 1] * 0.1, \
                                 mycell[0, 2] * 0.1, \
                                 mycell[1, 2] * 0.1))
        fileobj.write('END\n')
    return
