"""This module contains functions to read from QBox output files"""

from ase import Atom, Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import basestring

import re
import xml.etree.ElementTree as ET


# Compile regexs for fixing XML
re_find_bad_xml = re.compile(r'<(/?)([A-z]+) expectation ([a-z]+)')


def read_qbox(f, index=-1):
    """Read data from QBox output file

    Inputs:
        f - str or fileobj, path to file or file object to read from
        index - int or slice, which frames to return
    Returns:
        list of Atoms or atoms, requested frame(s)
    """

    if isinstance(f, basestring):
        f = open(f, 'r')

    # Check whether this is a QB@all output
    version = None
    for line in f:
        if '<release>' in line:
            version = ET.fromstring(line)
            break
    if version is None:
        raise Exception('Parse Error: Version not found')
    is_qball = 'qb@LL' in version.text or 'qball' in version.text

    # Load in atomic species
    species = dict()
    if is_qball:
        # Read all of the lines between release and the first call to `run`
        species_data = []
        for line in f:
            if '<run' in line:
                break
            species_data.append(line)
        species_data = '\n'.join(species_data)

        # Read out the species information with regular expressions
        symbols = re.findall('symbol_ = ([A-Z][a-z]?)', species_data)
        masses = re.findall('mass_ = ([0-9.]+)', species_data)
        names = re.findall('name_ = ([a-z]+)', species_data)
        numbers = re.findall('atomic_number_ = ([0-9]+)', species_data)

        # Compile them into a dictionary
        for name, symbol, mass, number in zip(names, symbols, masses, numbers):
            spec_data = dict(
                symbol=symbol,
                mass=float(mass),
                number=float(number)
            )
            species[name] = spec_data
    else:
        # Find all species
        species_blocks = _find_blocks(f, 'species', '<cmd>run')

        for spec in species_blocks:
            name = spec.get('name')
            spec_data = dict(
                symbol=spec.find('symbol').text,
                mass=float(spec.find('mass').text),
                number=int(spec.find('atomic_number').text))
            species[name] = spec_data

    # Find all of the frames
    frames = _find_blocks(f, 'iteration', None)

    # If index is an int, return one frame
    if isinstance(index, int):
        return _parse_frame(frames[index], species)
    else:
        return [_parse_frame(frame, species) for frame in frames[index]]


def _find_blocks(fp, tag, stopwords='[qbox]'):
    """Find and parse a certain block of the file.

    Reads a file sequentially and stops when it either encounters the end of the file, or until the it encounters a line
    that contains a user-defined string *after it has already found at least one desired block*. Use the stopwords
    ``[qbox]`` to read until the next command is issued.

    Groups the text between the first line that contains <tag> and the next line that contains </tag>, inclusively. The
    function then parses the XML and returns the Element object.

    Inputs:
        fp - file-like object, file to be read from
        tag - str, tag to search for (e.g., 'iteration'). `None` if you want to read until the end of the file
        stopwords - str, halt parsing if a line containing this string is encountered

    Returns:
        list of xml.ElementTree, parsed XML blocks found by this class
    """

    start_tag = '<%s'%tag
    end_tag = '</%s>'%tag

    blocks = []  # Stores all blocks
    cur_block = []  # Block being filled
    in_block = False  # Whether we are currently parsing
    for line in fp:

        # Check if the block has started
        if start_tag in line:
            if in_block:
                raise Exception('Parsing failed: Encountered nested block')
            else:
                in_block = True

        # Add data to block
        if in_block:
            cur_block.append(line)

        # Check for stopping conditions
        if stopwords is not None:
            if stopwords in line and len(blocks) > 0:
                break

        if end_tag in line:
            if in_block:
                blocks.append(cur_block)
                cur_block = []
                in_block = False
            else:
                raise Exception('Parsing failed: End tag found before start tag')

    # Join strings in a block into a single string
    blocks = [''.join(b) for b in blocks]

    # Ensure XML compatibility. There are two specific tags in QBall that are not
    #  valid XML, so we need to run a
    blocks = [re_find_bad_xml.sub(r'<\1\2_expectation_\3', b) for b in blocks]

    # Parse the blocks
    return [ET.fromstring(b) for b in blocks]


def _parse_frame(tree, species):
    """Parse a certain frame from QBOX output

    Inputs:
        tree - ElementTree, <iteration> block from output file
        species - dict, data about species. Key is name of atom type,
            value is data about that type
    Return:
        Atoms object describing this iteration"""

    # Load in data about the system
    energy = float(tree.find("etotal").text)

    # Load in data about the cell
    unitcell = tree.find('atomset').find('unit_cell')
    cell = []
    for d in ['a', 'b', 'c']:
        cell.append([float(x) for x in unitcell.get(d).split()])

    stress_tree = tree.find('stress_tensor')
    if stress_tree is None:
        stresses = None
    else:
        stresses = [float(stress_tree.find('sigma_%s' % x).text)
                    for x in ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']]

    # Create the Atoms object
    atoms = Atoms(pbc=True, cell=cell)

    # Load in the atom information
    forces = []
    for atom in tree.find('atomset').findall('atom'):
        # Load data about the atom type
        spec = atom.get('species')
        symbol = species[spec]['symbol']
        mass = species[spec]['mass']

        # Get data about position / velocity / force
        pos = [float(x) for x in atom.find('position').text.split()]
        force = [float(x) for x in atom.find('force').text.split()]
        momentum = [float(x) * mass
                    for x in atom.find('velocity').text.split()]

        # Create the objects
        atom = Atom(symbol=symbol, mass=mass, position=pos, momentum=momentum)
        atoms += atom
        forces.append(force)

    # Create the calculator object that holds energy/forces
    calc = SinglePointCalculator(atoms,
                                 energy=energy, forces=forces, stress=stresses)
    atoms.set_calculator(calc)

    return atoms
