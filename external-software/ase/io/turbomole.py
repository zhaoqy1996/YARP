from ase.units import Bohr


def read_turbomole(fd):
    """Method to read turbomole coord file

    coords in bohr, atom types in lowercase, format:
    $coord
    x y z atomtype
    x y z atomtype f
    $end
    Above 'f' means a fixed atom.
    """
    from ase import Atoms
    from ase.constraints import FixAtoms

    lines = fd.readlines()
    atoms_pos = []
    atom_symbols = []
    myconstraints=[]

    # find $coord section;
    # does not necessarily have to be the first $<something> in file...
    for i, l in enumerate(lines):
        if l.strip().startswith('$coord'):
            start = i
            break
    for line in lines[start+1:]:
        if line.startswith('$'): # start of new section
            break
        else:
            x, y, z, symbolraw = line.split()[:4]
            symbolshort=symbolraw.strip()
            symbol=symbolshort[0].upper()+symbolshort[1:].lower()
            #print symbol
            atom_symbols.append(symbol)
            atoms_pos.append([float(x)*Bohr, float(y)*Bohr, float(z)*Bohr])
            cols = line.split()
            if (len(cols) == 5):
                fixedstr = line.split()[4].strip()
                if (fixedstr == "f"):
                    myconstraints.append(True)
                else:
                    myconstraints.append(False)
            else:
                myconstraints.append(False)

    atoms = Atoms(positions = atoms_pos, symbols = atom_symbols, pbc = False)
    c = FixAtoms(mask = myconstraints)
    atoms.set_constraint(c)
    return atoms


def read_turbomole_gradient(fd, index=-1):
    """ Method to read turbomole gradient file """

    # read entire file
    lines = [x.strip() for x in fd.readlines()]

    # find $grad section
    start = end = -1
    for i, line in enumerate(lines):
        if not line.startswith('$'):
            continue
        if line.split()[0] == '$grad':
            start = i
        elif start >= 0:
            end = i
            break

    if end <= start:
        raise RuntimeError('File does not contain a valid \'$grad\' section')

    def formatError():
        raise RuntimeError('Data format in file does not correspond to known '
                           'Turbomole gradient format')

    # trim lines to $grad
    del lines[:start+1]
    del lines[end-1-start:]

    # Interpret $grad section
    from ase import Atoms, Atom
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.units import Bohr, Hartree
    images = []
    while len(lines): # loop over optimization cycles
        # header line
        # cycle =      1    SCF energy =     -267.6666811409   |dE/dxyz| =  0.157112
        fields = lines[0].split('=')
        try:
            # cycle = int(fields[1].split()[0])
            energy = float(fields[2].split()[0]) * Hartree
            # gradient = float(fields[3].split()[0])
        except (IndexError, ValueError):
            formatError()

        # coordinates/gradient
        atoms = Atoms()
        forces = []
        for line in lines[1:]:
            fields = line.split()
            if len(fields) == 4: # coordinates
                # 0.00000000000000      0.00000000000000      0.00000000000000      c
                try:
                    symbol = fields[3].lower().capitalize()
                    position = tuple([Bohr * float(x) for x in fields[0:3] ])
                except ValueError:
                    formatError()
                atoms.append(Atom(symbol, position))
            elif len(fields) == 3: # gradients
                #  -.51654903354681D-07  -.51654903206651D-07  0.51654903169644D-07
                try:
                    grad = [-float(x.replace('D', 'E')) * Hartree / Bohr for x in fields[0:3] ]
                except ValueError:
                    formatError()
                forces.append(grad)
            else: # next cycle
                break

        # calculator
        calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        atoms.set_calculator(calc)

        # save frame
        images.append(atoms)

        # delete this frame from data to be handled
        del lines[:2*len(atoms)+1]

    return images[index]


def write_turbomole(fd, atoms):
    """ Method to write turbomole coord file
    """
    from ase.constraints import FixAtoms

    coord = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    fix_indices = set()
    if atoms.constraints:
        for constr in atoms.constraints:
            if isinstance(constr, FixAtoms):
                fix_indices.update(constr.get_indices())

    fix_str = []
    for i in range(len(atoms)):
        if i in fix_indices:
            fix_str.append('f')
        else:
            fix_str.append('')

    fd.write('$coord\n')
    for (x, y, z), s, fix in zip(coord, symbols, fix_str):
        fd.write('%20.14f  %20.14f  %20.14f      %2s  %2s \n'
                 % (x / Bohr, y / Bohr, z / Bohr, s.lower(), fix))

    fd.write('$end\n')
