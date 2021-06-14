"""
SHELX (.res) input/output

Read/write files in SHELX (.res) file format.

Format documented at http://shelx.uni-ac.gwdg.de/SHELX/

Written by Martin Uhren and Georg Schusteritsch.
Adapted for ASE by James Kermode.
"""

from __future__ import division

import glob
import re

from ase.atoms import Atoms
from ase.geometry import cellpar_to_cell, cell_to_cellpar
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator

__all__ = ['Res', 'read_res', 'write_res']


class Res(object):

    """
    Object for representing the data in a Res file.
    Most attributes can be set directly.

    Args:
        atoms (Atoms):  Atoms object.

    .. attribute:: atoms

        Associated Atoms object.

    .. attribute:: name

        The name of the structure.

    .. attribute:: pressure

        The external pressure.

    .. attribute:: energy

        The internal energy of the structure.

    .. attribute:: spacegroup

        The space group of the structure.

    .. attribute:: times_found

        The number of times the structure was found.
    """

    def __init__(self, atoms, name=None, pressure=None,
                 energy=None, spacegroup=None, times_found=None):
        self.atoms_ = atoms
        if name is None:
            name = atoms.info.get('name')
        if pressure is None:
            pressure = atoms.info.get('pressure')
        if spacegroup is None:
            spacegroup = atoms.info.get('spacegroup')
        if times_found is None:
            times_found = atoms.info.get('times_found')
        self.name = name
        self.pressure = pressure
        self.energy = energy
        self.spacegroup = spacegroup
        self.times_found = times_found

    @property
    def atoms(self):
        """
        Returns Atoms object associated with this Res.
        """
        return self.atoms_

    @staticmethod
    def from_file(filename):
        """
        Reads a Res from a file.

        Args:
            filename (str): File name containing Res data.

        Returns:
            Res object.
        """
        with open(filename, 'r') as f:
            return Res.from_string(f.read())

    @staticmethod
    def parse_title(line):
        info = dict()

        tokens = line.split()
        num_tokens = len(tokens)
        # 1 = Name
        if num_tokens <= 1:
            return info
        info['name'] = tokens[1]
        # 2 = Pressure
        if num_tokens <= 2:
            return info
        info['pressure'] = float(tokens[2])
        # 3 = Volume
        # 4 = Internal energy
        if num_tokens <= 4:
            return info
        info['energy'] = float(tokens[4])
        # 5 = Spin density, 6 - Abs spin density
        # 7 = Space group OR num atoms (new format ONLY)
        idx = 7
        if tokens[idx][0] != '(':
            idx += 1

        if num_tokens <= idx:
            return info
        info['spacegroup'] = tokens[idx][1:len(tokens[idx]) - 1]
        # idx + 1 = n, idx + 2 = -
        # idx + 3 = times found
        if num_tokens <= idx + 3:
            return info
        info['times_found'] = int(tokens[idx + 3])

        return info

    @staticmethod
    def from_string(data):
        """
        Reads a Res from a string.

        Args:
            data (str): string containing Res data.

        Returns:
            Res object.
        """
        abc = []
        ang = []
        sp = []
        coords = []
        info = dict()
        coord_patt = re.compile(r"""(\w+)\s+
                                    ([0-9]+)\s+
                                    ([0-9\-\.]+)\s+
                                    ([0-9\-\.]+)\s+
                                    ([0-9\-\.]+)\s+
                                    ([0-9\-\.]+)""", re.VERBOSE)
        lines = data.splitlines()
        line_no = 0
        while line_no < len(lines):
            line = lines[line_no]
            tokens = line.split()
            if tokens:
                if tokens[0] == 'TITL':
                    try:
                        info = Res.parse_title(line)
                    except (ValueError, IndexError):
                        info = dict()
                elif tokens[0] == 'CELL' and len(tokens) == 8:
                    abc = [float(tok) for tok in tokens[2:5]]
                    ang = [float(tok) for tok in tokens[5:8]]
                elif tokens[0] == 'SFAC':
                    for atom_line in lines[line_no:]:
                        if line.strip() == 'END':
                            break
                        else:
                            match = coord_patt.search(atom_line)
                            if match:
                                sp.append(match.group(1))  # 1-indexed
                                cs = match.groups()[2:5]
                                coords.append([float(c) for c in cs])
                        line_no += 1  # Make sure the global is updated
            line_no += 1

        return Res(Atoms(symbols=sp,
                         scaled_positions=coords,
                         cell=cellpar_to_cell(list(abc) + list(ang)),
                         pbc=True, info=info),
                   info.get('name'),
                   info.get('pressure'),
                   info.get('energy'),
                   info.get('spacegroup'),
                   info.get('times_found'))

    def get_string(self, significant_figures=6, write_info=False):
        """
        Returns a string to be written as a Res file.

        Args:
            significant_figures (int): No. of significant figures to
                output all quantities. Defaults to 6.

            write_info (bool): if True, format TITL line using key-value pairs
               from atoms.info in addition to attributes stored in Res object

        Returns:
            String representation of Res.
        """

        # Title line
        if write_info:
            info = self.atoms.info.copy()
            for attribute in ['name', 'pressure', 'energy',
                              'spacegroup', 'times_found']:
                if getattr(self, attribute) and attribute not in info:
                    info[attribute] = getattr(self, attribute)
            lines = ['TITL ' + ' '.join(['{0}={1}'.format(k, v)
                                         for (k, v) in info.items()])]
        else:
            lines = ['TITL ' + self.print_title()]

        # Cell
        abc_ang = cell_to_cellpar(self.atoms.get_cell())
        fmt = '{{0:.{0}f}}'.format(significant_figures)
        cell = ' '.join([fmt.format(a) for a in abc_ang])
        lines.append('CELL 1.0 ' + cell)

        # Latt
        lines.append('LATT -1')

        # Atoms
        symbols = self.atoms.get_chemical_symbols()
        species_types = []
        for symbol in symbols:
            if symbol not in species_types:
                species_types.append(symbol)
        lines.append('SFAC ' + ' '.join(species_types))

        fmt = '{{0}} {{1}} {{2:.{0}f}} {{3:.{0}f}} {{4:.{0}f}} 1.0'
        fmtstr = fmt.format(significant_figures)
        for symbol, coords in zip(symbols,
                                  self.atoms_.get_scaled_positions()):
            lines.append(
                fmtstr.format(symbol,
                              species_types.index(symbol) + 1,
                              coords[0],
                              coords[1],
                              coords[2]))
        lines.append('END')
        return '\n'.join(lines)

    def __str__(self):
        """
        String representation of Res file.
        """
        return self.get_string()

    def write_file(self, filename, **kwargs):
        """
        Writes Res to a file. The supported kwargs are the same as those for
        the Res.get_string method and are passed through directly.
        """
        with open(filename, 'w') as f:
            f.write(self.get_string(**kwargs) + '\n')

    def print_title(self):
        tokens = [self.name, self.pressure, self.atoms.get_volume(),
                  self.energy, 0.0, 0.0, len(self.atoms)]
        if self.spacegroup:
            tokens.append('(' + self.spacegroup + ')')
        else:
            tokens.append('(P1)')
        if self.times_found:
            tokens.append('n - ' + str(self.times_found))
        else:
            tokens.append('n - 1')

        return ' '.join([str(tok) for tok in tokens])


def read_res(filename, index=-1):
    """
    Read input in SHELX (.res) format

    Multiple frames are read if `filename` contains a wildcard character,
    e.g. `file_*.res`. `index` specifes which frames to retun: default is
    last frame only (index=-1).
    """
    images = []
    for fn in sorted(glob.glob(filename)):
        res = Res.from_file(fn)
        if res.energy:
            calc = SinglePointCalculator(res.atoms,
                                         energy=res.energy)
            res.atoms.set_calculator(calc)
        images.append(res.atoms)
    return images[index]


def write_res(filename, images, write_info=True,
              write_results=True, significant_figures=6):
    """
    Write output in SHELX (.res) format

    To write multiple images, include a % format string in filename,
    e.g. `file_%03d.res`.

    Optionally include contents of Atoms.info dictionary if `write_info`
    is True, and/or results from attached calculator if `write_results`
    is True (only energy results are supported).
    """

    if not isinstance(images, (list, tuple)):
        images = [images]

    if len(images) > 1 and '%' not in filename:
        raise RuntimeError('More than one Atoms provided but no %' +
                           ' format string found in filename')

    for i, atoms in enumerate(images):
        fn = filename
        if '%' in filename:
            fn = filename % i
        res = Res(atoms)
        if write_results:
            calculator = atoms.get_calculator()
            if (calculator is not None and
                    isinstance(calculator, Calculator)):
                energy = calculator.results.get('energy')
                if energy is not None:
                    res.energy = energy
        res.write_file(fn, write_info=write_info,
                       significant_figures=significant_figures)
