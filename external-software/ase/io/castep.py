from __future__ import print_function
# -*- coding: utf-8 -*-
"""This module defines I/O routines with CASTEP files.
The key idea is that all function accept or return  atoms objects.
CASTEP specific parameters will be returned through the <atoms>.calc
attribute.
"""
import os
import re
import warnings
import numpy as np
from copy import deepcopy

import ase

from ase.parallel import paropen
from ase.spacegroup import Spacegroup
from ase.geometry.cell import cellpar_to_cell
from ase.constraints import FixAtoms, FixedPlane, FixedLine, FixCartesian

# independent unit management included here:
# When high accuracy is required, this allows to easily pin down
# unit conversion factors from different "unit definition systems"
# (CODATA1986 for ase-3.6.0.2515 vs CODATA2002 for CASTEP 5.01).
#
# ase.units in in ase-3.6.0.2515 is based on CODATA1986
import ase.units
units_ase = {
    'hbar': ase.units._hbar * ase.units.J,
    'Eh': ase.units.Hartree,
    'kB': ase.units.kB,
    'a0': ase.units.Bohr,
    't0': ase.units._hbar * ase.units.J / ase.units.Hartree,
    'c': ase.units._c,
    'me': ase.units._me / ase.units._amu,
    'Pascal': 1.0 / ase.units.Pascal}

# CODATA1986 (included herein for the sake of completeness)
# taken from
#    http://physics.nist.gov/cuu/Archive/1986RMP.pdf
units_CODATA1986 = {
    'hbar': 6.5821220E-16,      # eVs
    'Eh': 27.2113961,           # eV
    'kB': 8.617385E-5,          # eV/K
    'a0': 0.529177249,          # A
    'c': 299792458,             # m/s
    'e': 1.60217733E-19,        # C
    'me': 5.485799110E-4}       # u

# CODATA2002: default in CASTEP 5.01
# (-> check in more recent CASTEP in case of numerical discrepancies?!)
# taken from
#    http://physics.nist.gov/cuu/Document/all_2002.pdf
units_CODATA2002 = {
    'hbar': 6.58211915E-16,     # eVs
    'Eh': 27.2113845,           # eV
    'kB': 8.617343E-5,          # eV/K
    'a0': 0.5291772108,         # A
    'c': 299792458,             # m/s
    'e': 1.60217653E-19,        # C
    'me': 5.4857990945E-4}      # u

# (common) derived entries
for d in (units_CODATA1986, units_CODATA2002):
    d['t0'] = d['hbar'] / d['Eh']     # s
    d['Pascal'] = d['e'] * 1E30       # Pa


__all__ = [
    # routines for the generic io function
    'read_castep',
    'read_castep_castep',
    'read_castep_castep_old',
    'read_cell',
    'read_castep_cell',
    'read_geom',
    'read_castep_geom',
    'read_phonon',
    'read_castep_phonon',
    # additional reads that still need to be wrapped
    'read_md',
    'read_param',
    'read_seed',
    # write that is already wrapped
    'write_castep_cell',
    # param write - in principle only necessary in junction with the calculator
    'write_param']


def write_freeform(fd, outputobj):
    """
    Prints out to a given file a CastepInputFile or derived class, such as
    CastepCell or CastepParam.
    """

    options = outputobj._options

    # Some keywords, if present, are printed in this order
    preferred_order = ['lattice_cart', 'lattice_abc',
                       'positions_frac', 'positions_abs',
                       'species_pot', 'symmetry_ops',   # CELL file
                       'task', 'cut_off_energy'         # PARAM file
                       ]

    keys = outputobj.get_attr_dict().keys()
    # This sorts only the ones in preferred_order and leaves the rest
    # untouched
    keys = sorted(keys, key=lambda x: preferred_order.index(x)
                  if x in preferred_order
                  else len(preferred_order))

    for kw in keys:
        opt = options[kw]
        if opt.type.lower() == 'block':
            fd.write('%BLOCK {0}\n{1}\n%ENDBLOCK {0}\n\n'.format(
                     kw.upper(),
                     opt.value.strip('\n')))
        else:
            fd.write('{0}: {1}\n'.format(kw.upper(), opt.value))


def write_cell(filename, atoms, positions_frac=False, castep_cell=None,
               force_write=False):
    """
    Wrapper function for the more generic write() functionality.

    Note that this is function is intended to maintain backwards-compatibility
    only.
    """
    from ase.io import write

    write(filename, atoms, positions_frac=positions_frac,
          castep_cell=castep_cell, force_write=force_write)


def write_castep_cell(fd, atoms, positions_frac=False, force_write=False,
                      precision=6, magnetic_moments=None,
                      castep_cell=None):
    """
    This CASTEP export function write minimal information to
    a .cell file. If the atoms object is a trajectory, it will
    take the last image.

    Note that function has been altered in order to require a filedescriptor
    rather than a filename. This allows to use the more generic write()
    function from formats.py

    Note that the "force_write" keywords has no effect currently.

    Arguments:

        positions_frac: boolean. If true, positions are printed as fractional
                        rather than absolute. Default is false.
        castep_cell: if provided, overrides the existing CastepCell object in
                     the Atoms calculator
        precision: number of digits to which lattice and positions are printed
        magnetic_moments: if None, no SPIN values are initialised.
                          If 'initial', the values from
                          get_initial_magnetic_moments() are used.
                          If 'calculated', the values from 
                          get_magnetic_moments() are used.
                          If an array of the same length as the atoms object,
                          its contents will be used as magnetic moments.
    """

    if atoms is None:
        print('Atoms object not initialized')
        return False
    if isinstance(atoms, list):
        if len(atoms) > 1:
            atoms = atoms[-1]

    # Header
    fd.write('#######################################################\n')
    fd.write('#CASTEP cell file: %s\n' % fd.name)
    fd.write('#Created using the Atomic Simulation Environment (ASE)#\n')
    fd.write('#######################################################\n\n')

    # To write this we simply use the existing Castep calculator, or create
    # one
    from ase.calculators.castep import Castep, CastepCell

    try:
        has_cell = isinstance(atoms.calc.cell, CastepCell)
    except AttributeError:
        has_cell = False

    if has_cell:
        cell = deepcopy(atoms.calc.cell)
    else:
        cell = Castep(keyword_tolerance=2).cell

    # Write lattice
    fformat = '%{0}.{1}f'.format(precision+3, precision)
    cell_block_format = ' '.join([fformat]*3)
    cell.lattice_cart = [cell_block_format % tuple(line)
                         for line in atoms.get_cell()]

    if positions_frac:
        pos_keyword = 'positions_frac'
        positions = atoms.get_scaled_positions()
    else:
        pos_keyword = 'positions_abs'
        positions = atoms.get_positions()

    if atoms.has('castep_custom_species'):
        elems = atoms.get_array('castep_custom_species')
    else:
        elems = atoms.get_chemical_symbols()

    if atoms.has('castep_labels'):
        labels = atoms.get_array('castep_labels')
    else:
        labels = ['NULL']*len(elems)

    if str(magnetic_moments).lower() == 'initial':
        magmoms = atoms.get_initial_magnetic_moments()
    elif str(magnetic_moments).lower() == 'calculated':
        magmoms = atoms.get_magnetic_moments()
    elif np.array(magnetic_moments).shape == (len(elems),):
        magmoms = np.array(magnetic_moments)
    else:
        magmoms = [0]*len(elems)

    pos_block = []
    pos_block_format = '%s ' + cell_block_format

    for i, el in enumerate(elems):
        xyz = positions[i]
        line = pos_block_format % tuple([el] + list(xyz))
        # ADD other keywords if necessary
        if magmoms[i] != 0:
            line += ' SPIN={0} '.format(magmoms[i])
        if labels[i].strip() not in ('NULL', ''):
            line += ' LABEL={0} '.format(labels[i])
        pos_block.append(line)

    setattr(cell, pos_keyword, pos_block)

    constraints = atoms.constraints
    if len(constraints):
        _supported_constraints = (FixAtoms, FixedPlane, FixedLine,
                                  FixCartesian)

        constr_block = []

        for constr in constraints:
            if not isinstance(constr, _supported_constraints):
                print('Warning: you have constraints in your atoms, that are')
                print('         not supported by the CASTEP ase interface')
                break
            if isinstance(constr, FixAtoms):
                for i in constr.index:

                    try:
                        symbol = atoms.get_chemical_symbols()[i]
                        nis = atoms.calc._get_number_in_species(i)
                    except KeyError:
                        raise UserWarning('Unrecognized index in' +
                                          ' constraint %s' % constr)
                    for j in range(3):
                        l = '%6d %3s %3d   ' % (len(constr_block)+1,
                                                symbol,
                                                nis)
                        l += ['1 0 0', '0 1 0', '0 0 1'][j]
                        constr_block += [l]

            elif isinstance(constr, FixCartesian):
                n = constr.a
                symbol = atoms.get_chemical_symbols()[n]
                nis = atoms.calc._get_number_in_species(n)

                for i, m in enumerate(constr.mask):
                    if m == 1:
                        continue
                    l = '%6d %3s %3d   ' % (len(constr_block)+1, symbol, nis)
                    l += ' '.join(['1' if j == i else '0' for j in range(3)])
                    constr_block += [l]

            elif isinstance(constr, FixedPlane):
                n = constr.a
                symbol = atoms.get_chemical_symbols()[n]
                nis = atoms.calc._get_number_in_species(n)

                l = '%6d %3s %3d   ' % (len(constr_block)+1, symbol, nis)
                l += ' '.join([str(d) for d in constr.dir])
                constr_block += [l]

            elif isinstance(constr, FixedLine):
                n = constr.a
                symbol = atoms.get_chemical_symbols()[n]
                nis = atoms.calc._get_number_in_species(n)

                direction = constr.dir
                ((i1, v1), (i2, v2)) = sorted(enumerate(direction),
                                              key=lambda x: abs(x[1]),
                                              reverse=True)[:2]
                n1 = np.zeros(3)
                n1[i2] = v1
                n1[i1] = -v2
                n1 = n1 / np.linalg.norm(n1)

                n2 = np.cross(direction, n1)

                l1 = '%6d %3s %3d   %f %f %f' % (len(constr_block)+1,
                                                 symbol, nis,
                                                 n1[0], n1[1], n1[2])
                l2 = '%6d %3s %3d   %f %f %f' % (len(constr_block)+2,
                                                 symbol, nis,
                                                 n2[0], n2[1], n2[2])

                constr_block += [l1, l2]

        cell.ionic_constraints = constr_block

    write_freeform(fd, cell)

    return True


def read_freeform(fd):
    """
    Read a CASTEP freeform file (the basic format of .cell and .param files)
    and return keyword-value pairs as a dict (values are strings for single
    keywords and lists of strings for blocks).
    """

    from ase.calculators.castep import CastepInputFile

    inputobj = CastepInputFile(keyword_tolerance=2)

    filelines = fd.readlines()

    keyw = None
    read_block = False
    block_lines = None

    for i, l in enumerate(filelines):

        # Strip all comments, aka anything after a hash
        l = re.split(r'[#!;]', l, 1)[0].strip()

        if l == '':
            # Empty line... skip
            continue

        lsplit = re.split(r'\s*[:=]*\s+', l, 1)

        if read_block:
            if lsplit[0].lower() == '%endblock':
                if len(lsplit) == 1 or lsplit[1].lower() != keyw:
                    raise ValueError('Out of place end of block at '
                                     'line %i in freeform file' % i+1)
                else:
                    read_block = False
                    inputobj.__setattr__(keyw, block_lines)
            else:
                block_lines += [l]
        else:
            # Check the first word

            # Is it a block?
            read_block = (lsplit[0].lower() == '%block')
            if read_block:
                if len(lsplit) == 1:
                    raise ValueError(('Unrecognizable block at line %i '
                                      'in io freeform file') % i+1)
                else:
                    keyw = lsplit[1].lower()
            else:
                keyw = lsplit[0].lower()

            # Now save the value
            if read_block:
                block_lines = []
            else:
                inputobj.__setattr__(keyw, ' '.join(lsplit[1:]))

    return inputobj.get_attr_dict()


def read_cell(filename, index=None):
    """
    Wrapper function for the more generic read() functionality.

    Note that this is function is intended to maintain backwards-compatibility
    only.
    """
    from ase.io import read
    return read(filename, index=index, format='castep-cell')


def read_castep_cell(fd, index=None, calculator_args={}, find_spg=False,
                     units=units_CODATA2002):
    """Read a .cell file and return an atoms object.
    Any value found that does not fit the atoms API
    will be stored in the atoms.calc attribute.

    By default, the Castep calculator will be tolerant and in the absence of a
    castep_keywords.json file it will just accept all keywords that aren't 
    automatically parsed.
    """

    from ase.calculators.castep import Castep

    cell_units = {  # Units specifiers for CASTEP
        'bohr': units_CODATA2002['a0'],
        'ang': 1.0,
        'm': 1e10,
        'cm': 1e8,
        'nm': 10,
        'pm': 1e-2
    }

    calc = Castep(**calculator_args)

    if calc.cell.castep_version == 0 and calc._kw_tol < 3:
        # No valid castep_keywords.json was found
        print('read_cell: Warning - Was not able to validate CASTEP input.')
        print('           This may be due to a non-existing '
              '"castep_keywords.json"')
        print('           file or a non-existing CASTEP installation.')
        print('           Parsing will go on but keywords will not be '
              'validated and may cause problems if incorrect during a CASTEP '
              'run.')

    celldict = read_freeform(fd)

    def parse_blockunit(line_tokens, blockname):
        u = 1.0
        if len(line_tokens[0]) == 1:
            usymb = line_tokens[0][0].lower()
            u = cell_units.get(usymb, 1)
            if usymb not in cell_units:
                warnings.warn(('read_cell: Warning - ignoring invalid '
                               'unit specifier in %BLOCK {0} '
                               '(assuming Angstrom instead)'
                               ).format(blockname))
            line_tokens = line_tokens[1:]
        return u, line_tokens

    # Arguments to pass to the Atoms object at the end
    aargs = {
        'pbc': True
    }

    # Start by looking for the lattice
    lat_keywords = [w in celldict for w in ('lattice_cart', 'lattice_abc')]
    if all(lat_keywords):
        warnings.warn('read_cell: Warning - two lattice blocks present in the'
                      ' same file. LATTICE_ABC will be ignored')
    elif not any(lat_keywords):
        raise ValueError('Cell file must contain at least one between '
                         'LATTICE_ABC and LATTICE_CART')

    if 'lattice_abc' in celldict:

        lines = celldict.pop('lattice_abc').split('\n')
        line_tokens = [l.split() for l in lines]

        u, line_tokens = parse_blockunit(line_tokens, 'lattice_abc')

        if len(line_tokens) != 2:
            warnings.warn('read_cell: Warning - ignoring additional '
                          'lines in invalid %BLOCK LATTICE_ABC')

        abc = [float(p) * u for p in line_tokens[0][:3]]
        angles = [float(phi) for phi in line_tokens[1][:3]]

        aargs['cell'] = cellpar_to_cell(abc + angles)

    if 'lattice_cart' in celldict:

        lines = celldict.pop('lattice_cart').split('\n')
        line_tokens = [l.split() for l in lines]

        u, line_tokens = parse_blockunit(line_tokens, 'lattice_cart')

        if len(line_tokens) != 3:
            warnings.warn('read_cell: Warning - ignoring more than '
                          'three lattice vectors in invalid %BLOCK '
                          'LATTICE_CART')

        aargs['cell'] = [[float(x)*u for x in lt[:3]] for lt in line_tokens]

    # Now move on to the positions
    pos_keywords = [w in celldict
                    for w in ('positions_abs', 'positions_frac')]

    if all(pos_keywords):
        warnings.warn('read_cell: Warning - two lattice blocks present in the'
                      ' same file. POSITIONS_FRAC will be ignored')
        del celldict['positions_frac']
    elif not any(pos_keywords):
        raise ValueError('Cell file must contain at least one between '
                         'POSITIONS_FRAC and POSITIONS_ABS')

    aargs['symbols'] = []
    pos_type = 'positions'
    pos_block = celldict.pop('positions_abs', None)
    if pos_block is None:
        pos_type = 'scaled_positions'
        pos_block = celldict.pop('positions_frac', None)
    aargs[pos_type] = []

    lines = pos_block.split('\n')
    line_tokens = [l.split() for l in lines]

    if not 'scaled' in pos_type:
        u, line_tokens = parse_blockunit(line_tokens, 'positions_abs')
    else:
        u = 1.0

    # Here we extract all the possible additional info
    # These are marked by their type

    add_info = {
        'SPIN':   (float, 0.0),   # (type, default)
        'MAGMOM': (float, 0.0),
        'LABEL':  (str, 'NULL')
    }
    add_info_arrays = dict((k, []) for k in add_info)

    def parse_info(raw_info):

        re_keys = (r'({0})\s*[=:\s]{{1}}\s'
                   r'*([^\s]*)').format('|'.join(add_info.keys()))
        # Capture all info groups
        info = re.findall(re_keys, raw_info)
        info = {g[0]: add_info[g[0]][0](g[1]) for g in info}
        return info

    # Array for custom species (a CASTEP special thing)
    # Usually left unused
    custom_species = None

    for tokens in line_tokens:
        # Now, process the whole 'species' thing
        spec_custom = tokens[0].split(':', 1)
        elem = spec_custom[0]
        if len(spec_custom) > 1 and custom_species is None:
            # Add it to the custom info!
            custom_species = list(aargs['symbols'])
        if custom_species is not None:
            custom_species.append(tokens[0])
        aargs['symbols'].append(elem)
        aargs[pos_type].append([float(p) * u for p in tokens[1:4]])
        # Now for the additional information
        info = ' '.join(tokens[4:])
        info = parse_info(info)
        for k in add_info:
            add_info_arrays[k] += [info.get(k, add_info[k][1])]

    # Now on to the species potentials...
    if 'species_pot' in celldict:
        lines = celldict.pop('species_pot').split('\n')
        line_tokens = [l.split() for l in lines]

        for tokens in line_tokens:
            if len(tokens) == 1:
                # It's a library
                all_spec = (set(custom_species) if custom_species is not None
                            else set(aargs['symbols']))
                for s in all_spec:
                    calc.cell.species_pot = (s, tokens[0])
            else:
                calc.cell.species_pot = tuple(tokens[:2])

    # Ionic constraints
    raw_constraints = {}

    if 'ionic_constraints' in celldict:
        lines = celldict.pop('ionic_constraints').split('\n')
        line_tokens = [l.split() for l in lines]

        for tokens in line_tokens:
            if not len(tokens) == 6:
                continue
            _, species, nic, x, y, z = tokens
            # convert xyz to floats
            x = float(x)
            y = float(y)
            z = float(z)

            nic = int(nic)
            if (species, nic) not in raw_constraints:
                raw_constraints[(species, nic)] = []
            raw_constraints[(species, nic)].append(np.array(
                                                   [x, y, z]))

    # Symmetry operations
    if 'symmetry_ops' in celldict:
        lines = celldict.pop('symmetry_ops').split('\n')
        line_tokens = [l.split() for l in lines]

        # Read them in blocks of four
        blocks = np.array(line_tokens).astype(float)
        if (len(blocks.shape) != 2 or blocks.shape[1] != 3 or
                blocks.shape[0] % 4 != 0):
            warnings.warn('Warning: could not parse SYMMETRY_OPS'
                          ' block properly, skipping')
        else:
            blocks = blocks.reshape((-1, 4, 3))
            rotations = blocks[:, :3]
            translations = blocks[:, 3]

            # Regardless of whether we recognize them, store these
            calc.cell.symmetry_ops = (rotations, translations)

    # Anything else that remains, just add it to the cell object:
    for k, val in celldict.items():
        try:
            calc.cell.__setattr__(k, val)
        except Exception:
            raise RuntimeError('Problem setting calc.cell.%s = %s' % (k, val))

    # Get the relevant additional info
    aargs['magmoms'] = np.array(add_info_arrays['SPIN'])
    # SPIN or MAGMOM are alternative keywords
    aargs['magmoms'] = np.where(aargs['magmoms'] != 0,
                                aargs['magmoms'],
                                add_info_arrays['MAGMOM'])
    labels = np.array(add_info_arrays['LABEL'])

    aargs['calculator'] = calc

    atoms = ase.Atoms(**aargs)

    # Spacegroup...
    if find_spg:
        # Try importing spglib
        try:
            import spglib
        except ImportError:
            try:
                from pyspglib import spglib
            except ImportError:
                # spglib is not present
                warnings.warn('spglib not found installed on this system - '
                              'automatic spacegroup detection is not possible')
                spglib = None

        if spglib is not None:
            symmd = spglib.get_symmetry_dataset(atoms)
            atoms_spg = Spacegroup(int(symmd['number']))
            atoms.info['spacegroup'] = atoms_spg

    atoms.new_array('castep_labels', labels)
    if custom_species is not None:
        atoms.new_array('castep_custom_species', np.array(custom_species))

    fixed_atoms = []
    constraints = []
    for (species, nic), value in raw_constraints.items():
        absolute_nr = atoms.calc._get_absolute_number(species, nic)
        if len(value) == 3:
            # Check if they are linearly independent
            if np.linalg.det(value) == 0:
                print('Error: Found linearly dependent constraints attached '
                      'to atoms %s' % (absolute_nr))
                continue
            fixed_atoms.append(absolute_nr)
        elif len(value) == 2:
            direction = np.cross(value[0], value[1])
            # Check if they are linearly independent
            if np.linalg.norm(direction) == 0:
                print('Error: Found linearly dependent constraints attached '
                      'to atoms %s' % (absolute_nr))
                continue
            constraint = ase.constraints.FixedLine(
                a=absolute_nr,
                direction=direction)
            constraints.append(constraint)
        elif len(value) == 1:
            constraint = ase.constraints.FixedPlane(
                a=absolute_nr,
                direction=np.array(value[0], dtype=np.float32))
            constraints.append(constraint)
        else:
            print('Error: Found %s statements attached to atoms %s'
                  % (len(value), absolute_nr))

    # we need to sort the fixed atoms list in order not to raise an assertion
    # error in FixAtoms
    if fixed_atoms:
        constraints.append(
            ase.constraints.FixAtoms(indices=sorted(fixed_atoms)))
    if constraints:
        atoms.set_constraint(constraints)

    atoms.calc.atoms = atoms
    atoms.calc.push_oldstate()

    return atoms


def read_castep(filename, index=None):
    """
    Wrapper function for the more generic read() functionality.

    Note that this is function is intended to maintain backwards-compatibility
    only.
    """
    from ase.io import read
    return read(filename, index=index, format='castep-castep')


def read_castep_castep(fd, index=None):
    """
    Reads a .castep file and returns an atoms  object.
    The calculator information will be stored in the calc attribute.

    There is no use of the "index" argument as of now, it is just inserted for
    convenience to comply with the generic "read()" in ase.io

    Please note that this routine will return an atom ordering as found
    within the castep file. This means that the species will be ordered by
    ascending atomic numbers. The atoms witin a species are ordered as given
    in the original cell file.

    Note: This routine returns a single atoms_object only, the last
    configuration in the file. Yet, if you want to parse an MD run, use the
    novel function `read_md()`
    """

    from ase.calculators.castep import Castep

    try:
        calc = Castep()
    except Exception as e:
        # No CASTEP keywords found?
        print('WARNING:\n{0}\nUsing fallback .castep reader...'.format(e))
        # Fall back on the old method
        return read_castep_castep_old(fd, index)

    calc.read(castep_file=fd)

    # now we trick the calculator instance such that we can savely extract
    # energies and forces from this atom. Basically what we do is to trick the
    # internal routine calculation_required() to always return False such that
    # we do not need to re-run a CASTEP calculation.
    #
    # Probably we can solve this with a flag to the read() routine at some
    # point, but for the moment I do not want to change too much in there.
    calc._old_atoms = calc.atoms
    calc._old_param = calc.param
    calc._old_cell = calc.cell

    return [calc.atoms]  # Returning in the form of a list for next()


def read_castep_castep_old(fd, index=None):
    """
    DEPRECATED
    Now replaced by ase.calculators.castep.Castep.read(). Left in for future
    reference and backwards compatibility needs, as well as a fallback for
    when castep_keywords.py can't be created.

    Reads a .castep file and returns an atoms  object.
    The calculator information will be stored in the calc attribute.
    If more than one SCF step is found, a list of all steps
    will be stored in the traj attribute.

    Note that the index argument has no effect as of now.

    Please note that this routine will return an atom ordering as found
    within the castep file. This means that the species will be ordered by
    ascending atomic numbers. The atoms witin a species are ordered as given
    in the original cell file.
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    lines = fd.readlines()

    traj = []
    energy_total = None
    energy_0K = None
    for i, line in enumerate(lines):
        if 'NB est. 0K energy' in line:
            energy_0K = float(line.split()[6])
        # support also for dispersion correction
        elif 'NB dispersion corrected est. 0K energy*' in line:
            energy_0K = float(line.split()[-2])
        elif 'Final energy, E' in line:
            energy_total = float(line.split()[4])
        elif 'Dispersion corrected final energy' in line:
            pass
            # dispcorr_energy_total = float(line.split()[-2])
            # sedc_apply = True
        elif 'Dispersion corrected final free energy' in line:
            pass  # dispcorr_energy_free = float(line.split()[-2])
        elif 'dispersion corrected est. 0K energy' in line:
            pass  # dispcorr_energy_0K = float(line.split()[-2])
        elif 'Unit Cell' in line:
            cell = [x.split()[0:3] for x in lines[i + 3:i + 6]]
            cell = np.array([[float(col) for col in row] for row in cell])
        elif 'Cell Contents' in line:
            geom_starts = i
            start_found = False
            for j, jline in enumerate(lines[geom_starts:]):
                if jline.find('xxxxx') > 0 and start_found:
                    geom_stop = j + geom_starts
                    break
                if jline.find('xxxx') > 0 and not start_found:
                    geom_start = j + geom_starts + 4
                    start_found = True
            species = [line.split()[1] for line in lines[geom_start:geom_stop]]
            geom = np.dot(np.array([[float(col) for col in line.split()[3:6]]
                                    for line in lines[geom_start:geom_stop]]),
                          cell)
        elif 'Writing model to' in line:
            atoms = ase.Atoms(
                cell=cell,
                pbc=True,
                positions=geom,
                symbols=''.join(species))
            # take 0K energy where available, else total energy
            if energy_0K:
                energy = energy_0K
            else:
                energy = energy_total
            # generate a minimal single-point calculator
            sp_calc = SinglePointCalculator(atoms=atoms,
                                            energy=energy,
                                            forces=None,
                                            magmoms=None,
                                            stress=None)
            atoms.set_calculator(sp_calc)
            traj.append(atoms)
    if index is None:
        return traj
    else:
        return traj[index]


def read_geom(filename, index=':', units=units_CODATA2002):
    """
    Wrapper function for the more generic read() functionality.

    Note that this is function is intended to maintain backwards-compatibility
    only. Keyword arguments will be passed to read_castep_geom().
    """
    from ase.io import read
    return read(filename, index=index, format='castep-geom', units=units)


def read_castep_geom(fd, index=None, units=units_CODATA2002):
    """Reads a .geom file produced by the CASTEP GeometryOptimization task and
    returns an atoms  object.
    The information about total free energy and forces of each atom for every
    relaxation step will be stored for further analysis especially in a
    single-point calculator.
    Note that everything in the .geom file is in atomic units, which has
    been conversed to commonly used unit angstrom(length) and eV (energy).

    Note that the index argument has no effect as of now.

    Contribution by Wei-Bing Zhang. Thanks!

    Routine now accepts a filedescriptor in order to out-source the *.gz and
    *.bz2 handling to formats.py. Note that there is a fallback routine
    read_geom() that behaves like previous versions did.
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    # fd is closed by embracing read() routine
    txt = fd.readlines()

    traj = []

    Hartree = units['Eh']
    Bohr = units['a0']

    # Yeah, we know that...
    # print('N.B.: Energy in .geom file is not 0K extrapolated.')
    for i, line in enumerate(txt):
        if line.find('<-- E') > 0:
            start_found = True
            energy = float(line.split()[0]) * Hartree
            cell = [x.split()[0:3] for x in txt[i + 1:i + 4]]
            cell = np.array([[float(col) * Bohr for col in row] for row in
                             cell])
        if line.find('<-- R') > 0 and start_found:
            start_found = False
            geom_start = i
            for i, line in enumerate(txt[geom_start:]):
                if line.find('<-- F') > 0:
                    geom_stop = i + geom_start
                    break
            species = [line.split()[0] for line in
                       txt[geom_start:geom_stop]]
            geom = np.array([[float(col) * Bohr for col in
                              line.split()[2:5]] for line in
                             txt[geom_start:geom_stop]])
            forces = np.array([[float(col) * Hartree / Bohr for col in
                                line.split()[2:5]] for line in
                               txt[geom_stop:geom_stop +
                                   (geom_stop - geom_start)]])
            image = ase.Atoms(species, geom, cell=cell, pbc=True)
            image.set_calculator(
                SinglePointCalculator(atoms=image, energy=energy,
                                      forces=forces))
            traj.append(image)

    if index is None:
        return traj
    else:
        return traj[index]


def read_phonon(filename, index=None, read_vib_data=False,
                gamma_only=True, frequency_factor=None,
                units=units_CODATA2002):
    """
    Wrapper function for the more generic read() functionality.

    Note that this is function is intended to maintain backwards-compatibility
    only. For documentation see read_castep_phonon().
    """
    from ase.io import read

    if read_vib_data:
        full_output = True
    else:
        full_output = False

    return read(filename, index=index, format='castep-phonon',
                full_output=full_output, read_vib_data=read_vib_data,
                gamma_only=gamma_only, frequency_factor=frequency_factor,
                units=units)


def read_castep_phonon(fd, index=None, read_vib_data=False,
                       gamma_only=True, frequency_factor=None,
                       units=units_CODATA2002):
    """
    Reads a .phonon file written by a CASTEP Phonon task and returns an atoms
    object, as well as the calculated vibrational data if requested.

    Note that the index argument has no effect as of now.
    """

    # fd is closed by embracing read() routine
    lines = fd.readlines()

    atoms = None
    cell = []
    N = Nb = Nq = 0
    scaled_positions = []
    symbols = []
    masses = []

    # header
    l = 0
    while l < len(lines):

        line = lines[l]

        if 'Number of ions' in line:
            N = int(line.split()[3])
        elif 'Number of branches' in line:
            Nb = int(line.split()[3])
        elif 'Number of wavevectors'in line:
            Nq = int(line.split()[3])
        elif 'Unit cell vectors (A)' in line:
            for ll in range(3):
                l += 1
                fields = lines[l].split()
                cell.append([float(x) for x in fields[0:3]])
        elif 'Fractional Co-ordinates' in line:
            for ll in range(N):
                l += 1
                fields = lines[l].split()
                scaled_positions.append([float(x) for x in fields[1:4]])
                symbols.append(fields[4])
                masses.append(float(fields[5]))
        elif 'END header' in line:
            l += 1
            atoms = ase.Atoms(symbols=symbols,
                              scaled_positions=scaled_positions,
                              cell=cell)
            break

        l += 1

    # Eigenmodes and -vectors
    if frequency_factor is None:
        Kayser_to_eV = 1E2 * 2 * np.pi * units['hbar'] * units['c']
    # N.B. "fixed default" unit for frequencies in .phonon files is "cm-1"
    # (i.e. the latter is unaffected by the internal unit conversion system of
    # CASTEP!) set conversion factor to convert therefrom to eV by default for
    # now
    frequency_factor = Kayser_to_eV
    qpoints = []
    weights = []
    frequencies = []
    displacements = []
    for nq in range(Nq):
        fields = lines[l].split()
        qpoints.append([float(x) for x in fields[2:5]])
        weights.append(float(fields[5]))
    freqs = []
    for ll in range(Nb):
        l += 1
        fields = lines[l].split()
        freqs.append(frequency_factor * float(fields[1]))
    frequencies.append(np.array(freqs))

    # skip the two Phonon Eigenvectors header lines
    l += 2

    # generate a list of displacements with a structure that is identical to
    # what is stored internally in the Vibrations class (see in
    # ase.vibrations.Vibrations.modes):
    #      np.array(displacements).shape == (Nb,3*N)

    disps = []
    for ll in range(Nb):
        disp_coords = []
        for lll in range(N):
            l += 1
            fields = lines[l].split()
            disp_x = float(fields[2]) + float(fields[3]) * 1.0j
            disp_y = float(fields[4]) + float(fields[5]) * 1.0j
            disp_z = float(fields[6]) + float(fields[7]) * 1.0j
            disp_coords.extend([disp_x, disp_y, disp_z])
        disps.append(np.array(disp_coords))
    displacements.append(np.array(disps))

    if read_vib_data:
        if gamma_only:
            vibdata = [frequencies[0], displacements[0]]
        else:
            vibdata = [qpoints, weights, frequencies, displacements]
        return vibdata, atoms
    else:
        return atoms


def read_md(filename, index=None, return_scalars=False,
            units=units_CODATA2002):
    """Wrapper function for the more generic read() functionality.

    Note that this function is intended to maintain backwards-compatibility
    only. For documentation see read_castep_md()
    """
    if return_scalars:
        full_output = True
    else:
        full_output = False

    from ase.io import read
    return read(filename, index=index, format='castep-md',
                full_output=full_output, return_scalars=return_scalars,
                units=units)


def read_castep_md(fd, index=None, return_scalars=False,
                   units=units_CODATA2002):
    """Reads a .md file written by a CASTEP MolecularDynamics task
    and returns the trajectory stored therein as a list of atoms object.

    Note that the index argument has no effect as of now."""

    from ase.calculators.singlepoint import SinglePointCalculator

    factors = {
        't': units['t0'] * 1E15,     # fs
        'E': units['Eh'],            # eV
        'T': units['Eh'] / units['kB'],
        'P': units['Eh'] / units['a0']**3 * units['Pascal'],
        'h': units['a0'],
        'hv': units['a0'] / units['t0'],
        'S': units['Eh'] / units['a0']**3,
        'R': units['a0'],
        'V': np.sqrt(units['Eh'] / units['me']),
        'F': units['Eh'] / units['a0']}

    # fd is closed by embracing read() routine
    lines = fd.readlines()

    l = 0
    while 'END header' not in lines[l]:
        l += 1
    l_end_header = l
    lines = lines[l_end_header + 1:]
    times = []
    energies = []
    temperatures = []
    pressures = []
    traj = []

    # Initialization
    time = None
    Epot = None
    Ekin = None
    EH = None
    temperature = None
    pressure = None
    symbols = None
    positions = None
    cell = None
    velocities = None
    symbols = []
    positions = []
    velocities = []
    forces = []
    cell = np.eye(3)
    cell_velocities = []
    stress = []

    for (l, line) in enumerate(lines):
        fields = line.split()
        if len(fields) == 0:
            if l != 0:
                times.append(time)
                energies.append([Epot, EH, Ekin])
                temperatures.append(temperature)
                pressures.append(pressure)
                atoms = ase.Atoms(symbols=symbols,
                                  positions=positions,
                                  cell=cell)
                atoms.set_velocities(velocities)
                if len(stress) == 0:
                    atoms.set_calculator(
                        SinglePointCalculator(atoms=atoms, energy=Epot,
                                              forces=forces))
                else:
                    atoms.set_calculator(
                        SinglePointCalculator(atoms=atoms, energy=Epot,
                                              forces=forces, stress=stress))
                traj.append(atoms)
            symbols = []
            positions = []
            velocities = []
            forces = []
            cell = []
            cell_velocities = []
            stress = []
            continue
        if len(fields) == 1:
            time = factors['t'] * float(fields[0])
            continue

        if fields[-1] == 'E':
            E = [float(x) for x in fields[0:3]]
            Epot, EH, Ekin = [factors['E'] * Ei for Ei in E]
            continue

        if fields[-1] == 'T':
            temperature = factors['T'] * float(fields[0])
            continue

        # only printed in case of variable cell calculation or calculate_stress
        # explicitly requested
        if fields[-1] == 'P':
            pressure = factors['P'] * float(fields[0])
            continue
        if fields[-1] == 'h':
            h = [float(x) for x in fields[0:3]]
            cell.append([factors['h'] * hi for hi in h])
            continue

        # only printed in case of variable cell calculation
        if fields[-1] == 'hv':
            hv = [float(x) for x in fields[0:3]]
            cell_velocities.append([factors['hv'] * hvi for hvi in hv])
            continue

        # only printed in case of variable cell calculation
        if fields[-1] == 'S':
            S = [float(x) for x in fields[0:3]]
            stress.append([factors['S'] * Si for Si in S])
            continue
        if fields[-1] == 'R':
            symbols.append(fields[0])
            R = [float(x) for x in fields[2:5]]
            positions.append([factors['R'] * Ri for Ri in R])
            continue
        if fields[-1] == 'V':
            V = [float(x) for x in fields[2:5]]
            velocities.append([factors['V'] * Vi for Vi in V])
            continue
        if fields[-1] == 'F':
            F = [float(x) for x in fields[2:5]]
            forces.append([factors['F'] * Fi for Fi in F])
            continue

    if index is None:
        pass
    else:
        traj = traj[index]

    if return_scalars:
        data = [times, energies, temperatures, pressures]
        return data, traj
    else:
        return traj


# Routines that only the calculator requires

def read_param(filename='', calc=None, fd=None, get_interface_options=False):

    if fd is None:
        if filename == '':
            raise ValueError('One between filename and fd must be provided')
        fd = open(filename)
    elif filename is not '':
        warnings.warn('Filestream used to read param, file name will be '
                      'ignored')

    # If necessary, get the interface options
    if get_interface_options:
        int_opts = {}
        optre = re.compile(r'# ASE_INTERFACE ([^\s]+) : ([^\s]+)')

        lines = fd.readlines()
        fd.seek(0)

        for l in lines:
            m = optre.search(l)
            if m:
                int_opts[m.groups()[0]] = m.groups()[1]

    data = read_freeform(fd)

    if calc is None:
        from ase.calculators.castep import Castep
        calc = Castep(check_castep_version=False, keyword_tolerance=2)

    for kw, val in data.items():
        calc.param.__setattr__(kw, val)

    if not get_interface_options:
        return calc
    else:
        return calc, int_opts


def write_param(filename, param, check_checkfile=False,
                force_write=False,
                interface_options=None):
    """Writes a CastepParam object to a CASTEP .param file

    Parameters:
        filename: the location of the file to write to. If it
        exists it will be overwritten without warning. If it
        doesn't it will be created.
        param: a CastepParam instance
        check_checkfile : if set to True, write_param will
        only write continuation or reuse statement
        if a restart file exists in the same directory
    """
    if os.path.isfile(filename) and not force_write:
        print('ase.io.castep.write_param: Set optional argument')
        print('force_write=True to overwrite %s.' % filename)
        return False

    out = paropen(filename, 'w')
    out.write('#######################################################\n')
    out.write('#CASTEP param file: %s\n' % filename)
    out.write('#Created using the Atomic Simulation Environment (ASE)#\n')
    if interface_options is not None:
        out.write('# Internal settings of the calculator\n')
        out.write('# This can be switched off by settings\n')
        out.write('# calc._export_settings = False\n')
        out.write('# If stated, this will be automatically processed\n')
        out.write('# by ase.io.castep.read_seed()\n')
        for option, value in sorted(interface_options.items()):
            out.write('# ASE_INTERFACE %s : %s\n' % (option, value))
    out.write('#######################################################\n\n')

    if check_checkfile:
        param = deepcopy(param)  # To avoid modifying the parent one
        for checktype in ['continuation', 'reuse']:
            opt = getattr(param, checktype)
            if opt and opt.value:
                fname = opt.value
                if fname == 'default':
                    fname = os.path.splitext(filename)[0] + '.check'
                if not (os.path.exists(fname) or
                        # CASTEP also understands relative path names, hence
                        # also check relative to the param file directory
                        os.path.exists(
                    os.path.join(os.path.dirname(filename),
                                 opt.value))):
                    opt.clear()

    write_freeform(out, param)

    out.close()


def read_seed(seed, new_seed=None, ignore_internal_keys=False):
    """A wrapper around the CASTEP Calculator in conjunction with
    read_cell and read_param. Basically this can be used to reuse
    a previous calculation which results in a triple of
    cell/param/castep file. The label of the calculation if pre-
    fixed with `copy_of_` and everything else will be recycled as
    much as possible from the addressed calculation.

    Please note that this routine will return an atoms ordering as specified
    in the cell file! It will thus undo the potential reordering internally
    done by castep.
    """

    directory = os.path.abspath(os.path.dirname(seed))
    seed = os.path.basename(seed)

    paramfile = os.path.join(directory, '%s.param' % seed)
    cellfile = os.path.join(directory, '%s.cell' % seed)
    castepfile = os.path.join(directory, '%s.castep' % seed)
    checkfile = os.path.join(directory, '%s.check' % seed)

    atoms = read_cell(cellfile)
    atoms.calc._directory = directory
    atoms.calc._rename_existing_dir = False
    atoms.calc._castep_pp_path = directory
    atoms.calc.merge_param(paramfile,
                           ignore_internal_keys=ignore_internal_keys)
    if new_seed is None:
        atoms.calc._label = 'copy_of_%s' % seed
    else:
        atoms.calc._label = str(new_seed)
    if os.path.isfile(castepfile):
        # _set_atoms needs to be True here
        # but we set it right back to False
        # atoms.calc._set_atoms = False
        # BUGFIX: I do not see a reason to do that!
        atoms.calc.read(castepfile)
        # atoms.calc._set_atoms = False

        # if here is a check file, we also want to re-use this information
        if os.path.isfile(checkfile):
            atoms.calc._check_file = os.path.basename(checkfile)

        # sync the top-level object with the
        # one attached to the calculator
        atoms = atoms.calc.atoms
    else:
        # There are cases where we only want to restore a calculator/atoms
        # setting without a castep file...
        pass
        # No print statement required in these cases
        print('Corresponding *.castep file not found.')
        print('Atoms object will be restored from *.cell and *.param only.')
    atoms.calc.push_oldstate()

    return atoms
