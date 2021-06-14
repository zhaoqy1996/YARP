
def read_elk(filename):
    """Import ELK atoms definition.

    Reads unitcell, atom positions, magmoms from elk.in/GEOMETRY.OUT file.
    """

    from ase import Atoms
    from ase.units import Bohr
    import numpy as np

    atoms = Atoms()
    fd = open(filename, 'r')
    lines = fd.readlines()
    fd.close()
    scale = np.ones(4)  # unit cell scale
    positions = []
    cell = []
    symbols = []
    magmoms = []
    periodic = np.array([True, True, True])
    # find cell scale
    for n, line in enumerate(lines):
        if line.split() == []:
            continue
        if line.strip() == 'scale':
            scale[0] = float(lines[n + 1])
        elif line.startswith('scale'):
            scale[int(line.strip()[-1])] = float(lines[n + 1])
    for n, line in enumerate(lines):
        if line.split() == []:
            continue
        if line.startswith('avec'):
            cell = np.array(
                [[float(v) * scale[1] for v in lines[n + 1].split()],
                 [float(v) * scale[2] for v in lines[n + 2].split()],
                 [float(v) * scale[3] for v in lines[n + 3].split()]])
        if line.startswith('atoms'):
            lines1 = lines[n + 1:]  # start subsearch
            spfname = []
            natoms = []
            atpos = []
            bfcmt = []
            for n1, line1 in enumerate(lines1):
                if line1.split() == []:
                    continue
                if 'spfname' in line1:
                    spfnamenow = lines1[n1].split()[0]
                    spfname.append(spfnamenow)
                    natomsnow = int(lines1[n1 + 1].split()[0])
                    natoms.append(natomsnow)
                    atposnow = []
                    bfcmtnow = []
                    for l in lines1[n1 + 2:n1 + 2 + natomsnow]:
                        atposnow.append([float(v) for v in l.split()[0:3]])
                        if len(l.split()) == 6:  # bfcmt present
                            bfcmtnow.append([float(v) for v in l.split()[3:]])
                    atpos.append(atposnow)
                    bfcmt.append(bfcmtnow)
    # symbols, positions, magmoms based on ELK spfname, atpos, and bfcmt
    symbols = ''
    positions = []
    magmoms = []
    for n, s in enumerate(spfname):
        symbols += str(s[1:].split('.')[0]) * natoms[n]
        positions += atpos[n]  # assumes fractional coordinates
        if len(bfcmt[n]) > 0:
            # how to handle cases of magmoms being one or three dim array?
            magmoms += [m[-1] for m in bfcmt[n]]
    atoms = Atoms(symbols, scaled_positions=positions, cell=[1, 1, 1])
    if len(magmoms) > 0:
        atoms.set_initial_magnetic_moments(magmoms)
    # final cell scale
    cell = cell * scale[0] * Bohr
    if periodic.any():
        atoms.set_cell(cell, scale_atoms=True)
        atoms.set_pbc(periodic)
    return atoms
