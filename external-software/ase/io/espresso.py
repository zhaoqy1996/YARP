"""Reads Quantum ESPRESSO files.

Read multiple structures and results from pw.x output files. Read
structures from pw.x input files.

Built for PWSCF v.5.3.0 but should work with earlier and later versions.
Can deal with most major functionality, but might fail with ibrav =/= 0
or crystal_sg positions.

Units are converted using CODATA 2006, as used internally by Quantum
ESPRESSO.
"""

import os
import operator as op
import warnings
from collections import OrderedDict
from os import path

import numpy as np

from ase.atoms import Atoms
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
                                         SinglePointKPoint)
from ase.calculators.calculator import kpts2ndarray, kpts2sizeandoffsets
from ase.dft.kpoints import kpoint_convert
from ase.constraints import FixAtoms, FixCartesian
from ase.data import chemical_symbols, atomic_numbers
from ase.units import create_units
from ase.utils import basestring


# Quantum ESPRESSO uses CODATA 2006 internally
units = create_units('2006')

# Section identifiers
_PW_START = 'Program PWSCF'
_PW_END = 'End of self-consistent calculation'
_PW_CELL = 'CELL_PARAMETERS'
_PW_POS = 'ATOMIC_POSITIONS'
_PW_MAGMOM = 'Magnetic moment per site'
_PW_FORCE = 'Forces acting on atoms'
_PW_TOTEN = '!    total energy'
_PW_STRESS = 'total   stress'
_PW_FERMI = 'the Fermi energy is'
_PW_KPTS = 'number of k points='
_PW_BANDS = 'End of '

class Namelist(OrderedDict):
    """Case insensitive dict that emulates Fortran Namelists."""
    def __contains__(self, key):
        return super(Namelist, self).__contains__(key.lower())

    def __delitem__(self, key):
        return super(Namelist, self).__delitem__(key.lower())

    def __getitem__(self, key):
        return super(Namelist, self).__getitem__(key.lower())

    def __setitem__(self, key, value):
        super(Namelist, self).__setitem__(key.lower(), value)

    def get(self, key, default=None):
        return super(Namelist, self).get(key.lower(), default)


def read_espresso_out(fileobj, index=-1, results_required=True):
    """Reads Quantum ESPRESSO output files.

    The atomistic configurations as well as results (energy, force, stress,
    magnetic moments) of the calculation are read for all configurations
    within the output file.

    Will probably raise errors for broken or incomplete files.

    Parameters
    ----------
    fileobj : file|str
        A file like object or filename
    index : slice
        The index of configurations to extract.
    results_required : bool
        If True, atomistic configurations that do not have any
        associated results will not be included. This prevents double
        printed configurations and incomplete calculations from being
        returned as the final configuration with no results data.

    Yields
    ------
    structure : Atoms
        The next structure from the index slice. The Atoms has a
        SinglePointCalculator attached with any results parsed from
        the file.


    """
    if isinstance(fileobj, basestring):
        fileobj = open(fileobj, 'rU')

    # work with a copy in memory for faster random access
    pwo_lines = fileobj.readlines()

    # TODO: index -1 special case?
    # Index all the interesting points
    indexes = {
        _PW_START: [],
        _PW_END: [],
        _PW_CELL: [],
        _PW_POS: [],
        _PW_MAGMOM: [],
        _PW_FORCE: [],
        _PW_TOTEN: [],
        _PW_STRESS: [],
        _PW_FERMI: [],
        _PW_KPTS: [],
        _PW_BANDS: [],
    }

    for idx, line in enumerate(pwo_lines):
        for identifier in indexes:
            if identifier in line:
                indexes[identifier].append(idx)

    # Configurations are either at the start, or defined in ATOMIC_POSITIONS
    # in a subsequent step. Can deal with concatenated output files.
    all_config_indexes = sorted(indexes[_PW_START] +
                                indexes[_PW_POS])

    # Slice only requested indexes
    # setting results_required argument stops configuration-only
    # structures from being returned. This ensures the [-1] structure
    # is one that has results. Two cases:
    # - SCF of last configuration is not converged, job terminated
    #   abnormally.
    # - 'relax' and 'vc-relax' re-prints the final configuration but
    #   only 'vc-relax' recalculates.
    if results_required:
        results_indexes = sorted(indexes[_PW_TOTEN] + indexes[_PW_FORCE] +
                                 indexes[_PW_STRESS] + indexes[_PW_MAGMOM] +
                                 indexes[_PW_BANDS])

        # Prune to only configurations with results data before the next
        # configuration
        results_config_indexes = []
        for config_index, config_index_next in zip(
                all_config_indexes,
                all_config_indexes[1:] + [len(pwo_lines)]):
            if any([config_index < results_index < config_index_next
                    for results_index in results_indexes]):
                results_config_indexes.append(config_index)

        # slice from the subset
        image_indexes = results_config_indexes[index]
    else:
        image_indexes = all_config_indexes[index]

    # Extract initialisation information each time PWSCF starts
    # to add to subsequent configurations. Use None so slices know
    # when to fill in the blanks.
    pwscf_start_info = dict((idx, None) for idx in indexes[_PW_START])

    for image_index in image_indexes:
        # Find the nearest calculation start to parse info. Needed in,
        # for example, relaxation where cell is only printed at the
        # start.
        if image_index in indexes[_PW_START]:
            prev_start_index = image_index
        else:
            # The greatest start index before this structure
            prev_start_index = [idx for idx in indexes[_PW_START]
                                if idx < image_index][-1]

        # add structure to reference if not there
        if pwscf_start_info[prev_start_index] is None:
            pwscf_start_info[prev_start_index] = parse_pwo_start(
                pwo_lines, prev_start_index)

        # Get the bounds for information for this structure. Any associated
        # values will be between the image_index and the following one,
        # EXCEPT for cell, which will be 4 lines before if it exists.
        for next_index in all_config_indexes:
            if next_index > image_index:
                break
        else:
            # right to the end of the file
            next_index = len(pwo_lines)

        # Get the structure
        # Use this for any missing data
        prev_structure = pwscf_start_info[prev_start_index]['atoms']
        if image_index in indexes[_PW_START]:
            structure = prev_structure.copy()  # parsed from start info
        else:
            if _PW_CELL in pwo_lines[image_index - 5]:
                # CELL_PARAMETERS would be just before positions if present
                cell, cell_alat = get_cell_parameters(
                    pwo_lines[image_index - 5:image_index])
            else:
                cell = prev_structure.cell
                cell_alat = pwscf_start_info[prev_start_index]['alat']

            # give at least enough lines to parse the positions
            # should be same format as input card
            n_atoms = len(prev_structure)
            positions_card = get_atomic_positions(
                pwo_lines[image_index:image_index + n_atoms + 1],
                n_atoms=n_atoms, cell=cell, alat=cell_alat)

            # convert to Atoms object
            symbols = [label_to_symbol(position[0]) for position in
                       positions_card]
            positions = [position[1] for position in positions_card]

            structure = Atoms(symbols=symbols, positions=positions, cell=cell,
                              pbc=True)

        # Extract calculation results
        # Energy
        energy = None
        for energy_index in indexes[_PW_TOTEN]:
            if image_index < energy_index < next_index:
                energy = float(
                    pwo_lines[energy_index].split()[-2]) * units['Ry']

        # Forces
        forces = None
        for force_index in indexes[_PW_FORCE]:
            if image_index < force_index < next_index:
                # Before QE 5.3 'negative rho' added 2 lines before forces
                # Use exact lines to stop before 'non-local' forces
                # in high verbosity
                if not pwo_lines[force_index + 2].strip():
                    force_index += 4
                else:
                    force_index += 2
                # assume contiguous
                forces = [
                    [float(x) for x in force_line.split()[-3:]] for force_line
                    in pwo_lines[force_index:force_index + len(structure)]]
                forces = np.array(forces) * units['Ry'] / units['Bohr']

        # Stress
        stress = None
        for stress_index in indexes[_PW_STRESS]:
            if image_index < stress_index < next_index:
                sxx, sxy, sxz = pwo_lines[stress_index + 1].split()[:3]
                _, syy, syz = pwo_lines[stress_index + 2].split()[:3]
                _, _, szz = pwo_lines[stress_index + 3].split()[:3]
                stress = np.array([sxx, syy, szz, syz, sxz, sxy], dtype=float)
                # sign convention is opposite of ase
                stress *= -1 * units['Ry'] / (units['Bohr'] ** 3)

        # Magmoms
        magmoms = None
        for magmoms_index in indexes[_PW_MAGMOM]:
            if image_index < magmoms_index < next_index:
                magmoms = [
                    float(mag_line.split()[5]) for mag_line
                    in pwo_lines[magmoms_index + 1:
                                 magmoms_index + 1 + len(structure)]]

        # Fermi level
        efermi = None
        for fermi_index in indexes[_PW_FERMI]:
            if image_index < fermi_index < next_index:
                efermi = float(pwo_lines[fermi_index].split()[-2])

        # K-points
        ibzkpts = None
        weights = None
        kpoints_warning = "Number of k-points >= 100: " + \
                          "set verbosity='high' to print them."

        for kpts_index in indexes[_PW_KPTS]:
            if image_index < kpts_index < next_index:
                nkpts = int(pwo_lines[kpts_index].split()[4])
                kpts_index += 2

                if pwo_lines[kpts_index].strip() == kpoints_warning:
                    continue

                # QE prints the k-points in units of 2*pi/alat
                # with alat defined as the length of the first
                # cell vector
                cell = structure.get_cell()
                alat = np.linalg.norm(cell[0])
                ibzkpts = []
                weights = []
                for i in range(nkpts):
                    l =  pwo_lines[kpts_index + i].split()
                    weights.append(float(l[-1]))
                    coord = map(float, [l[-6], l[-5], l[-4].strip('),')])
                    coord = np.array(coord) * 2 * np.pi / alat
                    coord = kpoint_convert(cell, ckpts_kv=coord)
                    ibzkpts.append(coord)
                ibzkpts = np.array(ibzkpts)
                weights = np.array(weights)

        # Bands
        kpts = None
        kpoints_warning = "Number of k-points >= 100: " + \
                          "set verbosity='high' to print the bands."

        for bands_index in indexes[_PW_BANDS]:
            if image_index < bands_index < next_index:
                bands_index += 2

                if pwo_lines[bands_index].strip() == kpoints_warning:
                    continue

                assert ibzkpts is not None
                spin, bands, eigenvalues = 0, [], [[], []]

                while True:
                    l = pwo_lines[bands_index].split()
                    if len(l) == 0:
                        if len(bands) > 0:
                            eigenvalues[spin].append(bands)
                            bands = []
                    elif l == ['occupation', 'numbers']:
                        bands_index += 3
                    elif l[0] == 'k' and l[1].startswith('='):
                        pass
                    elif len(l) > 2 and l[1] == 'SPIN':
                        if l[2] == 'DOWN':
                            spin += 1
                    else:
                        try:
                            bands.extend(map(float, l))
                        except ValueError:
                            break
                    bands_index += 1

                if spin == 1:
                    assert len(eigenvalues[0]) == len(eigenvalues[1])
                assert len(eigenvalues[0]) == len(ibzkpts)

                kpts = []
                for s in range(spin + 1):
                    for w, k, e in zip(weights, ibzkpts, eigenvalues[s]):
                        kpt = SinglePointKPoint(w, s, k, eps_n=e)
                        kpts.append(kpt)

        # Put everything together
        calc = SinglePointDFTCalculator(structure, energy=energy, 
                                        forces=forces, stress=stress, 
                                        magmoms=magmoms, efermi=efermi,
                                        ibzkpts=ibzkpts)
        calc.kpts = kpts
        structure.set_calculator(calc)

        yield structure


def parse_pwo_start(lines, index=0):
    """Parse Quantum ESPRESSO calculation info from lines,
    starting from index. Return a dictionary containing extracted
    information.

    - `celldm(1)`: lattice parameters (alat)
    - `cell`: unit cell in Angstrom
    - `symbols`: element symbols for the structure
    - `positions`: cartesian coordinates of atoms in Angstrom
    - `atoms`: an `ase.Atoms` object constructed from the extracted data

    Parameters
    ----------
    lines : list[str]
        Contents of PWSCF output file.
    index : int
        Line number to begin parsing. Only first calculation will
        be read.

    Returns
    -------
    info : dict
        Dictionary of calculation parameters, including `celldm(1)`, `cell`,
        `symbols`, `positions`, `atoms`.

    Raises
    ------
    KeyError
        If interdependent values cannot be found (especially celldm(1))
        an error will be raised as other quantities cannot then be
        calculated (e.g. cell and positions).
    """
    # TODO: extend with extra DFT info?

    info = {}

    for idx, line in enumerate(lines[index:], start=index):
        if 'celldm(1)' in line:
            # celldm(1) has more digits than alat!!
            info['celldm(1)'] = float(line.split()[1]) * units['Bohr']
            info['alat'] = info['celldm(1)']
        elif 'number of atoms/cell' in line:
            info['nat'] = int(line.split()[-1])
        elif 'number of atomic types' in line:
            info['ntyp'] = int(line.split()[-1])
        elif 'crystal axes:' in line:
            info['cell'] = info['celldm(1)'] * np.array([
                [float(x) for x in lines[idx + 1].split()[3:6]],
                [float(x) for x in lines[idx + 2].split()[3:6]],
                [float(x) for x in lines[idx + 3].split()[3:6]]])
        elif 'positions (alat units)' in line:
            info['symbols'] = [
                label_to_symbol(at_line.split()[1])
                for at_line in lines[idx + 1:idx + 1 + info['nat']]]
            info['positions'] = [
                [float(x) * info['celldm(1)'] for x in at_line.split()[6:9]]
                for at_line in lines[idx + 1:idx + 1 + info['nat']]]
            # This should be the end of interesting info.
            # Break here to avoid dealing with large lists of kpoints.
            # Will need to be extended for DFTCalculator info.
            break

    # Make atoms for convenience
    info['atoms'] = Atoms(symbols=info['symbols'],
                          positions=info['positions'],
                          cell=info['cell'], pbc=True)

    return info


def read_espresso_in(fileobj):
    """Parse a Quantum ESPRESSO input files, '.in', '.pwi'.

    ESPRESSO inputs are generally a fortran-namelist format with custom
    blocks of data. The namelist is parsed as a dict and an atoms object
    is constructed from the included information.

    Parameters
    ----------
    fileobj : file | str
        A file-like object that supports line iteration with the contents
        of the input file, or a filename.

    Returns
    -------
    atoms : Atoms
        Structure defined in the input file.

    Raises
    ------
    KeyError
        Raised for missing keys that are required to process the file
    """
    # TODO: use ase opening mechanisms
    if isinstance(fileobj, basestring):
        fileobj = open(fileobj, 'rU')

    # parse namelist section and extract remaining lines
    data, card_lines = read_fortran_namelist(fileobj)

    # get the cell if ibrav=0
    if 'system' not in data:
        raise KeyError('Required section &SYSTEM not found.')
    elif 'ibrav' not in data['system']:
        raise KeyError('ibrav is required in &SYSTEM')
    elif data['system']['ibrav'] == 0:
        # celldm(1) is in Bohr, A is in angstrom. celldm(1) will be
        # used even if A is also specified.
        if 'celldm(1)' in data['system']:
            alat = data['system']['celldm(1)'] * units['Bohr']
        elif 'A' in data['system']:
            alat = data['system']['A']
        else:
            alat = None
        cell, cell_alat = get_cell_parameters(card_lines, alat=alat)
    else:
        alat, cell = ibrav_to_cell(data['system'])

    positions_card = get_atomic_positions(
        card_lines, n_atoms=data['system']['nat'], cell=cell, alat=alat)

    symbols = [label_to_symbol(position[0]) for position in positions_card]
    positions = [position[1] for position in positions_card]

    # TODO: put more info into the atoms object
    # e.g magmom, force constraints
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    return atoms


def ibrav_to_cell(system):
    """
    Convert a value of ibrav to a cell. Any unspecified lattice dimension
    is set to 0.0, but will not necessarily raise an error. Also return the
    lattice parameter.

    Parameters
    ----------
    system : dict
        The &SYSTEM section of the input file, containing the 'ibrav' setting,
        and either celldm(1)..(6) or a, b, c, cosAB, cosAC, cosBC.

    Returns
    -------
    alat, cell : float, np.array
        Cell parameter in Angstrom, and
        The 3x3 array representation of the cell.

    Raises
    ------
    KeyError
        Raise an error if any required keys are missing.
    NotImplementedError
        Only a limited number of ibrav settings can be parsed. An error
        is raised if the ibrav interpretation is not implemented.
    """
    if 'celldm(1)' in system and 'a' in system:
        raise KeyError('do not specify both celldm and a,b,c!')
    elif 'celldm(1)' in system:
        # celldm(x) in bohr
        alat = system['celldm(1)'] * units['Bohr']
        b_over_a = system.get('celldm(2)', 0.0)
        c_over_a = system.get('celldm(3)', 0.0)
        cosab = system.get('celldm(4)', 0.0)
        cosac = system.get('celldm(5)', 0.0)
        cosbc = 0.0
        if system['ibrav'] == 14:
            cosbc = system.get('celldm(4)', 0.0)
            cosac = system.get('celldm(5)', 0.0)
            cosab = system.get('celldm(6)', 0.0)
    elif 'a' in system:
        # a, b, c, cosAB, cosAC, cosBC in Angstrom
        alat = system['a']
        b_over_a = system.get('b', 0.0) / alat
        c_over_a = system.get('c', 0.0) / alat
        cosab = system.get('cosab', 0.0)
        cosac = system.get('cosac', 0.0)
        cosbc = system.get('cosbc', 0.0)
    else:
        raise KeyError("Missing celldm(1) or a cell parameter.")

    if system['ibrav'] == 1:
        cell = np.identity(3) * alat
    elif system['ibrav'] == 2:
        cell = np.array([[-1.0, 0.0, 1.0],
                         [0.0, 1.0, 1.0],
                         [-1.0, 1.0, 0.0]]) * (alat / 2)
    elif system['ibrav'] == 3:
        cell = np.array([[1.0, 1.0, 1.0],
                         [-1.0, 1.0, 1.0],
                         [-1.0, -1.0, 1.0]]) * (alat / 2)
    elif system['ibrav'] == -3:
        cell = np.array([[-1.0, 1.0, 1.0],
                         [1.0, -1.0, 1.0],
                         [1.0, 1.0, -1.0]]) * (alat / 2)
    elif system['ibrav'] == 4:
        cell = np.array([[1.0, 0.0, 0.0],
                         [-0.5, 0.5*3**0.5, 0.0],
                         [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == 5:
        tx = ((1.0 - cosab) / 2.0)**0.5
        ty = ((1.0 - cosab) / 6.0)**0.5
        tz = ((1 + 2 * cosab) / 3.0)**0.5
        cell = np.array([[tx, -ty, tz],
                         [0, 2*ty, tz],
                         [-tx, -ty, tz]]) * alat
    elif system['ibrav'] == -5:
        ty = ((1.0 - cosab) / 6.0)**0.5
        tz = ((1 + 2 * cosab) / 3.0)**0.5
        a_prime = alat / 3**0.5
        u = tz - 2 * 2**0.5 * ty
        v = tz + 2**0.5 * ty
        cell = np.array([[u, v, v],
                         [v, u, v],
                         [v, v, u]]) * a_prime
    elif system['ibrav'] == 6:
        cell = np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == 7:
        cell = np.array([[1.0, -1.0, c_over_a],
                         [1.0, 1.0, c_over_a],
                         [-1.0, -1.0, c_over_a]]) * (alat / 2)
    elif system['ibrav'] == 8:
        cell = np.array([[1.0, 0.0, 0.0],
                         [0.0, b_over_a, 0.0],
                         [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == 9:
        cell = np.array([[1.0 / 2.0, b_over_a / 2.0, 0.0],
                         [-1.0 / 2.0, b_over_a / 2.0, 0.0],
                         [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == -9:
        cell = np.array([[1.0 / 2.0, -b_over_a / 2.0, 0.0],
                         [1.0 / 2.0, b_over_a / 2.0, 0.0],
                         [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == 10:
        cell = np.array([[1.0 / 2.0, 0.0, c_over_a/2.0],
                         [1.0 / 2.0, b_over_a / 2.0, 0.0],
                         [0.0, b_over_a / 2.0, c_over_a / 2.0]]) * alat
    elif system['ibrav'] == 11:
        cell = np.array([[1.0 / 2.0, b_over_a / 2.0, c_over_a / 2.0],
                         [-1.0 / 2.0, b_over_a / 2.0, c_over_a / 2.0],
                         [-1.0, 2.0, -b_over_a / 2.0, c_over_a / 2.0]]) * alat
    elif system['ibrav'] == 12:
        sinab = (1.0 - cosab**2)**0.5
        cell = np.array([[1.0, 0.0, 0.0],
                         [b_over_a * cosab, b_over_a * sinab, 0.0],
                         [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == -12:
        sinac = (1.0 - cosac**2)**0.5
        cell = np.array([[1.0, 0.0, 0.0],
                         [0.0, b_over_a, 0.0],
                         [c_over_a * cosac, 0.0, c_over_a * sinac]]) * alat
    elif system['ibrav'] == 13:
        sinab = (1.0 - cosab**2)**0.5
        cell = np.array([[1.0 / 2.0, 0.0, -c_over_a / 2.0],
                         [b_over_a * cosab, b_over_a * sinab, 0.0],
                         [1.0 / 2.0, 0.0, c_over_a / 2.0]]) * alat
    elif system['ibrav'] == 14:
        sinab = (1.0 - cosab**2)**0.5
        v3 = [c_over_a * cosac,
              c_over_a * (cosbc - cosac * cosab) / sinab,
              c_over_a * ((1 + 2 * cosbc * cosac * cosab
                           - cosbc**2 - cosac**2 - cosab**2)**0.5) / sinab]
        cell = np.array([[1.0, 0.0, 0.0],
                         [b_over_a * cosab, b_over_a * sinab, 0.0],
                         v3]) * alat
    else:
        raise NotImplementedError('ibrav = {0} is not implemented'
                                  ''.format(system['ibrav']))

    return alat, cell


def get_atomic_positions(lines, n_atoms, cell=None, alat=None):
    """Parse atom positions from ATOMIC_POSITIONS card.

    Parameters
    ----------
    lines : list[str]
        A list of lines containing the ATOMIC_POSITIONS card.
    n_atoms : int
        Expected number of atoms. Only this many lines will be parsed.
    cell : np.array
        Unit cell of the crystal. Only used with crystal coordinates.
    alat : float
        Lattice parameter for atomic coordinates. Only used for alat case.

    Returns
    -------
    positions : list[(str, (float, float, float), (float, float, float))]
        A list of the ordered atomic positions in the format:
        label, (x, y, z), (if_x, if_y, if_z)
        Force multipliers are set to None if not present.

    Raises
    ------
    ValueError
        Any problems parsing the data result in ValueError

    """

    positions = None
    # no blanks or comment lines, can the consume n_atoms lines for positions
    trimmed_lines = (line for line in lines
                     if line.strip() and not line[0] == '#')

    for line in trimmed_lines:
        if line.strip().startswith('ATOMIC_POSITIONS'):
            if positions is not None:
                raise ValueError('Multiple ATOMIC_POSITIONS specified')
            # Priority and behaviour tested with QE 5.3
            if 'crystal_sg' in line.lower():
                raise NotImplementedError('CRYSTAL_SG not implemented')
            elif 'crystal' in line.lower():
                cell = cell
            elif 'bohr' in line.lower():
                cell = np.identity(3) * units['Bohr']
            elif 'angstrom' in line.lower():
                cell = np.identity(3)
            # elif 'alat' in line.lower():
            #     cell = np.identity(3) * alat
            else:
                if alat is None:
                    raise ValueError('Set lattice parameter in &SYSTEM for '
                                     'alat coordinates')
                # Always the default, will be DEPRECATED as mandatory
                # in future
                cell = np.identity(3) * alat

            positions = []
            for _dummy in range(n_atoms):
                split_line = next(trimmed_lines).split()
                # These can be fractions and other expressions
                position = np.dot((infix_float(split_line[1]),
                                   infix_float(split_line[2]),
                                   infix_float(split_line[3])), cell)
                if len(split_line) > 4:
                    force_mult = (float(split_line[4]),
                                  float(split_line[5]),
                                  float(split_line[6]))
                else:
                    force_mult = None

                positions.append((split_line[0], position, force_mult))

    return positions


def get_cell_parameters(lines, alat=None):
    """Parse unit cell from CELL_PARAMETERS card.

    Parameters
    ----------
    lines : list[str]
        A list with lines containing the CELL_PARAMETERS card.
    alat : float | None
        Unit of lattice vectors in Angstrom. Only used if the card is
        given in units of alat. alat must be None if CELL_PARAMETERS card
        is in Bohr or Angstrom. For output files, alat will be parsed from
        the card header and used in preference to this value.

    Returns
    -------
    cell : np.array | None
        Cell parameters as a 3x3 array in Angstrom. If no cell is found
        None will be returned instead.
    cell_alat : float | None
        If a value for alat is given in the card header, this is also
        returned, otherwise this will be None.

    Raises
    ------
    ValueError
        If CELL_PARAMETERS are given in units of bohr or angstrom
        and alat is not
    """

    cell = None
    cell_alat = None
    # no blanks or comment lines, can take three lines for cell
    trimmed_lines = (line for line in lines
                     if line.strip() and not line[0] == '#')

    for line in trimmed_lines:
        if line.strip().startswith('CELL_PARAMETERS'):
            if cell is not None:
                # multiple definitions
                raise ValueError('CELL_PARAMETERS specified multiple times')
            # Priority and behaviour tested with QE 5.3
            if 'bohr' in line.lower():
                if alat is not None:
                    raise ValueError('Lattice parameters given in '
                                     '&SYSTEM celldm/A and CELL_PARAMETERS '
                                     'bohr')
                cell_units = units['Bohr']
            elif 'angstrom' in line.lower():
                if alat is not None:
                    raise ValueError('Lattice parameters given in '
                                     '&SYSTEM celldm/A and CELL_PARAMETERS '
                                     'angstrom')
                cell_units = 1.0
            elif 'alat' in line.lower():
                # Output file has (alat = value) (in Bohrs)
                if '=' in line:
                    alat = float(line.strip(') \n').split()[-1]) * units['Bohr']
                    cell_alat = alat
                elif alat is None:
                    raise ValueError('Lattice parameters must be set in '
                                     '&SYSTEM for alat units')
                cell_units = alat
            elif alat is None:
                # may be DEPRECATED in future
                cell_units = units['Bohr']
            else:
                # may be DEPRECATED in future
                cell_units = alat
            # Grab the parameters; blank lines have been removed
            cell = [[ffloat(x) for x in next(trimmed_lines).split()[:3]],
                    [ffloat(x) for x in next(trimmed_lines).split()[:3]],
                    [ffloat(x) for x in next(trimmed_lines).split()[:3]]]
            cell = np.array(cell) * cell_units

    return cell, cell_alat


def str_to_value(string):
    """Attempt to convert string into int, float (including fortran double),
    or bool, in that order, otherwise return the string.
    Valid (case-insensitive) bool values are: '.true.', '.t.', 'true'
    and 't' (or false equivalents).

    Parameters
    ----------
    string : str
        Test to parse for a datatype

    Returns
    -------
    value : any
        Parsed string as the most appropriate datatype of int, float,
        bool or string.

    """

    # Just an integer
    try:
        return int(string)
    except ValueError:
        pass
    # Standard float
    try:
        return float(string)
    except ValueError:
        pass
    # Fortran double
    try:
        return ffloat(string)
    except ValueError:
        pass

    # possible bool, else just the raw string
    if string.lower() in ('.true.', '.t.', 'true', 't'):
        return True
    elif string.lower() in ('.false.', '.f.', 'false', 'f'):
        return False
    else:
        return string.strip("'")


def read_fortran_namelist(fileobj):
    """Takes a fortran-namelist formatted file and returns nested
    dictionaries of sections and key-value data, followed by a list
    of lines of text that do not fit the specifications.

    Behaviour is taken from Quantum ESPRESSO 5.3. Parses fairly
    convoluted files the same way that QE should, but may not get
    all the MANDATORY rules and edge cases for very non-standard files:
        Ignores anything after '!' in a namelist, split pairs on ','
        to include multiple key=values on a line, read values on section
        start and end lines, section terminating character, '/', can appear
        anywhere on a line.
        All of these are ignored if the value is in 'quotes'.

    Parameters
    ----------
    fileobj : file
        An open file-like object.

    Returns
    -------
    data : dict of dict
        Dictionary for each section in the namelist with key = value
        pairs of data.
    card_lines : list of str
        Any lines not used to create the data, assumed to belong to 'cards'
        in the input file.

    """
    # Espresso requires the correct order
    data = Namelist()
    card_lines = []
    in_namelist = False
    section = 'none'  # can't be in a section without changing this

    for line in fileobj:
        # leading and trailing whitespace never needed
        line = line.strip()
        if line.startswith('&'):
            # inside a namelist
            section = line.split()[0][1:].lower()  # case insensitive
            if section in data:
                # Repeated sections are completely ignored.
                # (Note that repeated keys overwrite within a section)
                section = "_ignored"
            data[section] = Namelist()
            in_namelist = True
        if not in_namelist and line:
            # Stripped line is Truthy, so safe to index first character
            if line[0] not in ('!', '#'):
                card_lines.append(line)
        if in_namelist:
            # parse k, v from line:
            key = []
            value = None
            in_quotes = False
            for character in line:
                if character == ',' and value is not None and not in_quotes:
                    # finished value:
                    data[section][''.join(key).strip()] = str_to_value(
                        ''.join(value).strip())
                    key = []
                    value = None
                elif character == '=' and value is None and not in_quotes:
                    # start writing value
                    value = []
                elif character == "'":
                    # only found in value anyway
                    in_quotes = not in_quotes
                    value.append("'")
                elif character == '!' and not in_quotes:
                    break
                elif character == '/' and not in_quotes:
                    in_namelist = False
                    break
                elif value is not None:
                    value.append(character)
                else:
                    key.append(character)
            if value is not None:
                data[section][''.join(key).strip()] = str_to_value(
                    ''.join(value).strip())

    return data, card_lines


def ffloat(string):
    """Parse float from fortran compatible float definitions.

    In fortran exponents can be defined with 'd' or 'q' to symbolise
    double or quad precision numbers. Double precision numbers are
    converted to python floats and quad precision values are interpreted
    as numpy longdouble values (platform specific precision).

    Parameters
    ----------
    string : str
        A string containing a number in fortran real format

    Returns
    -------
    value : float | np.longdouble
        Parsed value of the string.

    Raises
    ------
    ValueError
        Unable to parse a float value.

    """

    if 'q' in string.lower():
        return np.longdouble(string.lower().replace('q', 'e'))
    else:
        return float(string.lower().replace('d', 'e'))


def label_to_symbol(label):
    """Convert a valid espresso ATOMIC_SPECIES label to a
    chemical symbol.

    Parameters
    ----------
    label : str
        chemical symbol X (1 or 2 characters, case-insensitive)
        or chemical symbol plus a number or a letter, as in
        "Xn" (e.g. Fe1) or "X_*" or "X-*" (e.g. C1, C_h;
        max total length cannot exceed 3 characters).

    Returns
    -------
    symbol : str
        The best matching species from ase.utils.chemical_symbols

    Raises
    ------
    KeyError
        Couldn't find an appropriate species.

    Notes
    -----
        It's impossible to tell whether e.g. He is helium
        or hydrogen labelled 'e'.
    """

    # possibly a two character species
    # ase Atoms need proper case of chemical symbols.
    if len(label) >= 2:
        test_symbol = label[0].upper() + label[1].lower()
        if test_symbol in chemical_symbols:
            return test_symbol
    # finally try with one character
    test_symbol = label[0].upper()
    if test_symbol in chemical_symbols:
        return test_symbol
    else:
        raise KeyError('Could not parse species from label {0}.'
                       ''.format(label))


def infix_float(text):
    """Parse simple infix maths into a float for compatibility with
    Quantum ESPRESSO ATOMIC_POSITIONS cards. Note: this works with the
    example, and most simple expressions, but the capabilities of
    the two parsers are not identical. Will also parse a normal float
    value properly, but slowly.

    >>> infix_float('1/2*3^(-1/2)')
    0.28867513459481287

    Parameters
    ----------
    text : str
        An arithmetic expression using +, -, *, / and ^, including brackets.

    Returns
    -------
    value : float
        Result of the mathematical expression.

    """

    def middle_brackets(full_text):
        """Extract text from innermost brackets."""
        start, end = 0, len(full_text)
        for (idx, char) in enumerate(full_text):
            if char == '(':
                start = idx
            if char == ')':
                end = idx + 1
                break
        return full_text[start:end]

    def eval_no_bracket_expr(full_text):
        """Calculate value of a mathematical expression, no brackets."""
        exprs = [('+', op.add), ('*', op.mul),
                 ('/', op.truediv), ('^', op.pow)]
        full_text = full_text.lstrip('(').rstrip(')')
        try:
            return float(full_text)
        except ValueError:
            for symbol, func in exprs:
                if symbol in full_text:
                    left, right = full_text.split(symbol, 1)  # single split
                    return func(eval_no_bracket_expr(left),
                                eval_no_bracket_expr(right))

    while '(' in text:
        middle = middle_brackets(text)
        text = text.replace(middle, '{}'.format(eval_no_bracket_expr(middle)))

    return float(eval_no_bracket_expr(text))

###
# Input file writing
###

# Ordered and case insensitive
KEYS = Namelist((
    ('CONTROL', [
        'calculation', 'title', 'verbosity', 'restart_mode', 'wf_collect',
        'nstep', 'iprint', 'tstress', 'tprnfor', 'dt', 'outdir', 'wfcdir',
        'prefix', 'lkpoint_dir', 'max_seconds', 'etot_conv_thr',
        'forc_conv_thr', 'disk_io', 'pseudo_dir', 'tefield', 'dipfield',
        'lelfield', 'nberrycyc', 'lorbm', 'lberry', 'gdir', 'nppstr',
        'lfcpopt', 'monopole']),
    ('SYSTEM', [
        'ibrav', 'celldm', 'A', 'B', 'C', 'cosAB', 'cosAC', 'cosBC', 'nat',
        'ntyp', 'nbnd', 'tot_charge', 'tot_magnetization',
        'starting_magnetization', 'ecutwfc', 'ecutrho', 'ecutfock', 'nr1',
        'nr2', 'nr3', 'nr1s', 'nr2s', 'nr3s', 'nosym', 'nosym_evc', 'noinv',
        'no_t_rev', 'force_symmorphic', 'use_all_frac', 'occupations',
        'one_atom_occupations', 'starting_spin_angle', 'degauss', 'smearing',
        'nspin', 'noncolin', 'ecfixed', 'qcutz', 'q2sigma', 'input_dft',
        'exx_fraction', 'screening_parameter', 'exxdiv_treatment',
        'x_gamma_extrapolation', 'ecutvcut', 'nqx1', 'nqx2', 'nqx3',
        'lda_plus_u', 'lda_plus_u_kind', 'Hubbard_U', 'Hubbard_J0',
        'Hubbard_alpha', 'Hubbard_beta', 'Hubbard_J',
        'starting_ns_eigenvalue', 'U_projection_type', 'edir',
        'emaxpos', 'eopreg', 'eamp', 'angle1', 'angle2',
        'constrained_magnetization', 'fixed_magnetization', 'lambda',
        'report', 'lspinorb', 'assume_isolated', 'esm_bc', 'esm_w',
        'esm_efield', 'esm_nfit', 'fcp_mu', 'vdw_corr', 'london',
        'london_s6', 'london_c6', 'london_rvdw', 'london_rcut',
        'ts_vdw_econv_thr', 'ts_vdw_isolated', 'xdm', 'xdm_a1', 'xdm_a2',
        'space_group', 'uniqueb', 'origin_choice', 'rhombohedral', 'zmon',
        'realxz', 'block', 'block_1', 'block_2', 'block_height']),
    ('ELECTRONS', [
        'electron_maxstep', 'scf_must_converge', 'conv_thr', 'adaptive_thr',
        'conv_thr_init', 'conv_thr_multi', 'mixing_mode', 'mixing_beta',
        'mixing_ndim', 'mixing_fixed_ns', 'diagonalization', 'ortho_para',
        'diago_thr_init', 'diago_cg_maxiter', 'diago_david_ndim',
        'diago_full_acc', 'efield', 'efield_cart', 'efield_phase',
        'startingpot', 'startingwfc', 'tqr']),
    ('IONS', [
        'ion_dynamics', 'ion_positions', 'pot_extrapolation',
        'wfc_extrapolation', 'remove_rigid_rot', 'ion_temperature', 'tempw',
        'tolp', 'delta_t', 'nraise', 'refold_pos', 'upscale', 'bfgs_ndim',
        'trust_radius_max', 'trust_radius_min', 'trust_radius_ini', 'w_1',
        'w_2']),
    ('CELL', [
        'cell_dynamics', 'press', 'wmass', 'cell_factor', 'press_conv_thr',
        'cell_dofree'])))


# Number of valence electrons in the pseudopotentials recommended by
# http://materialscloud.org/sssp/. These are just used as a fallback for
# calculating inital magetization values which are given as a fraction
# of valence electrons.
SSSP_VALENCE = [
    0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    18.0, 19.0, 20.0, 13.0, 14.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 12.0, 13.0, 14.0, 15.0, 6.0,
    7.0, 18.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
    19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 36.0, 27.0, 14.0, 15.0, 30.0,
    15.0, 32.0, 19.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0]


def construct_namelist(parameters=None, warn=False, **kwargs):
    """
    Construct an ordered Namelist containing all the parameters given (as
    a dictionary or kwargs). Keys will be inserted into their appropriate
    section in the namelist and the dictionary may contain flat and nested
    structures. Any kwargs that match input keys will be incorporated into
    their correct section. All matches are case-insensitive, and returned
    Namelist object is a case-insensitive dict.

    If a key is not known to ase, but in a section within `parameters`,
    it will be assumed that it was put there on purpose and included
    in the output namelist. Anything not in a section will be ignored (set
    `warn` to True to see ignored keys).

    Keys with a dimension (e.g. Hubbard_U(1)) will be incorporated as-is
    so the `i` should be made to match the output.

    The priority of the keys is:
        kwargs[key] > parameters[key] > parameters[section][key]
    Only the highest priority item will be included.

    Parameters
    ----------
    parameters: dict
        Flat or nested set of input parameters.
    warn: bool
        Enable warnings for unused keys.

    Returns
    -------
    input_namelist: Namelist
        pw.x compatible namelist of input parameters.

    """
    # Convert everything to Namelist early to make case-insensitive
    if parameters is None:
        parameters = Namelist()
    else:
        # Maximum one level of nested dict
        # Don't modify in place
        parameters_namelist = Namelist()
        for key, value in parameters.items():
            if isinstance(value, dict):
                parameters_namelist[key] = Namelist(value)
            else:
                parameters_namelist[key] = value
        parameters = parameters_namelist

    # Just a dict
    kwargs = Namelist(kwargs)

    # Final parameter set
    input_namelist = Namelist()

    # Collect
    for section in KEYS:
        sec_list = Namelist()
        for key in KEYS[section]:
            # Check all three separately and pop them all so that
            # we can check for missing values later
            if key in parameters.get(section, {}):
                sec_list[key] = parameters[section].pop(key)
            if key in parameters:
                sec_list[key] = parameters.pop(key)
            if key in kwargs:
                sec_list[key] = kwargs.pop(key)

            # Check if there is a key(i) version (no extra parsing)
            for arg_key in parameters.get(section, {}):
                if arg_key.split('(')[0].strip().lower() == key.lower():
                    sec_list[arg_key] = parameters[section].pop(arg_key)
            cp_parameters = parameters.copy()
            for arg_key in cp_parameters:
                if arg_key.split('(')[0].strip().lower() == key.lower():
                    sec_list[arg_key] = parameters.pop(arg_key)
            cp_kwargs = kwargs.copy()
            for arg_key in cp_kwargs:
                if arg_key.split('(')[0].strip().lower() == key.lower():
                    sec_list[arg_key] = kwargs.pop(arg_key)

        # Add to output
        input_namelist[section] = sec_list

    unused_keys = list(kwargs)
    # pass anything else already in a section
    for key, value in parameters.items():
        if key in KEYS and isinstance(value, dict):
            input_namelist[key].update(value)
        elif isinstance(value, dict):
            unused_keys.extend(list(value))
        else:
            unused_keys.append(key)

    if warn and unused_keys:
        warnings.warn('Unused keys: {}'.format(', '.join(unused_keys)))

    return input_namelist


def grep_valence(pseudopotential):
    """
    Given a UPF pseudopotential file, find the number of valence atoms.

    Parameters
    ----------
    pseudopotential: str
        Filename of the pseudopotential.

    Returns
    -------
    valence: float
        Valence as reported in the pseudopotential.

    Raises
    ------
    ValueError
        If valence cannot be found in the pseudopotential.
    """

    # Example lines
    # Sr.pbe-spn-rrkjus_psl.1.0.0.UPF:        z_valence="1.000000000000000E+001"
    # C.pbe-n-kjpaw_psl.1.0.0.UPF (new ld1.x):
    #                            ...PBC" z_valence="4.000000000000e0" total_p...
    # C_ONCV_PBE-1.0.upf:                     z_valence="    4.00"
    # Ta_pbe_v1.uspp.F.UPF:   13.00000000000      Z valence

    with open(pseudopotential) as psfile:
        for line in psfile:
            if 'z valence' in line.lower():
                return float(line.split()[0])
            elif 'z_valence' in line.lower():
                if line.split()[0] == '<PP_HEADER':
                    line = list(filter(lambda x: 'z_valence' in x,
                                       line.split(' ')))[0]
                return float(line.split('=')[-1].strip().strip('"'))
        else:
            raise ValueError('Valence missing in {}'.format(pseudopotential))


def cell_to_ibrav(cell, ibrav):
    """
    Calculate the appropriate `celldm(..)` parameters for the given ibrav
    using the given cell. The units for `celldm(..)` are Bohr.

    Does minimal checking of the cell shape, so it is possible to create
    a nonsense structure if the ibrav is inapproprite for the cell. These
    are derived to be symmetric with the routine for constructing the cell
    from ibrav parameters so directions of some vectors may be unexpected.

    Parameters
    ----------
    cell : np.array
        A 3x3 representation of a unit cell
    ibrav : int
        Bravais-lattice index according to the pw.x designations.

    Returns
    -------
    parameters : dict
        A dictionary with all the necessary `celldm(..)` keys assigned
        necessary values (in units of Bohr). Also includes `ibrav` so it
        can be passed back to `ibrav_to_cell`.

    Raises
    ------
    NotImplementedError
        Only a limited number of ibrav settings can be parsed. An error
        is raised if the ibrav interpretation is not implemented.
    """
    parameters = {'ibrav': ibrav}

    if ibrav == 1:
        parameters['celldm(1)'] = cell[0][0] / units['Bohr']
    elif ibrav in [2, 3, -3]:
        parameters['celldm(1)'] = cell[0][2] * 2 / units['Bohr']
    elif ibrav in [4, 6]:
        parameters['celldm(1)'] = cell[0][0] / units['Bohr']
        parameters['celldm(3)'] = cell[2][2] / cell[0][0]
    elif ibrav in [5, -5]:
        # Manually derive
        a = np.linalg.norm(cell[0])
        cosab = np.dot(cell[0], cell[1]) / (a ** 2)
        parameters['celldm(1)'] = a / units['Bohr']
        parameters['celldm(4)'] = cosab
    elif ibrav == 7:
        parameters['celldm(1)'] = cell[0][0] * 2 / units['Bohr']
        parameters['celldm(3)'] = cell[2][2] / cell[0][0]
    elif ibrav == 8:
        parameters['celldm(1)'] = cell[0][0] / units['Bohr']
        parameters['celldm(2)'] = cell[1][1] / cell[0][0]
        parameters['celldm(3)'] = cell[2][2] / cell[0][0]
    elif ibrav in [9, -9]:
        parameters['celldm(1)'] = cell[0][0] * 2 / units['Bohr']
        parameters['celldm(2)'] = cell[1][1] / cell[0][0]
        parameters['celldm(3)'] = cell[2][2] * 2 / cell[0][0]
    elif ibrav in [10, 11]:
        parameters['celldm(1)'] = cell[0][0] * 2 / units['Bohr']
        parameters['celldm(2)'] = cell[1][1] / cell[0][0]
        parameters['celldm(3)'] = cell[2][2] / cell[0][0]
    elif ibrav == 12:
        # cos^2 + sin^2
        b = (cell[1][0]**2 + cell[1][1]**2)**0.5
        parameters['celldm(1)'] = cell[0][0] / units['Bohr']
        parameters['celldm(2)'] = b / cell[0][0]
        parameters['celldm(3)'] = cell[2][2] / cell[0][0]
        parameters['celldm(4)'] = cell[1][0] / b
    elif ibrav == -12:
        # cos^2 + sin^2
        c = (cell[2][0]**2 + cell[2][2]**2)**0.5
        parameters['celldm(1)'] = cell[0][0] / units['Bohr']
        parameters['celldm(2)'] = cell[1][1] / cell[0][0]
        parameters['celldm(3)'] = c / cell[0][0]
        parameters['celldm(4)'] = cell[2][0] / c
    elif ibrav == 13:
        b = (cell[1][0]**2 + cell[1][1]**2)**0.5
        parameters['celldm(1)'] = cell[0][0] * 2 / units['Bohr']
        parameters['celldm(2)'] = b / (cell[0][0] * 2)
        parameters['celldm(3)'] = cell[2][2] / cell[0][0]
        parameters['celldm(4)'] = cell[1][0] / b
    elif ibrav == 14:
        # Manually derive
        a, b, c = np.linalg.norm(cell, axis=1)
        cosbc = np.dot(cell[1], cell[2]) / (b * c)
        cosac = np.dot(cell[0], cell[2]) / (a * c)
        cosab = np.dot(cell[0], cell[1]) / (a * b)
        parameters['celldm(1)'] = a / units['Bohr']
        parameters['celldm(2)'] = b / a
        parameters['celldm(3)'] = c / a
        parameters['celldm(4)'] = cosbc
        parameters['celldm(5)'] = cosac
        parameters['celldm(6)'] = cosab
    else:
        raise NotImplementedError('ibrav = {0} is not implemented'
                                  ''.format(ibrav))

    return parameters


def kspacing_to_grid(atoms, spacing, calculated_spacing=None):
    """
    Calculate the kpoint mesh that is equivalent to the given spacing
    in reciprocal space (units Angstrom^-1). The number of kpoints is each
    dimension is rounded up (compatible with CASTEP).

    Parameters
    ----------
    atoms: ase.Atoms
        A structure that can have get_reciprocal_cell called on it.
    spacing: float
        Minimum K-Point spacing in $A^{-1}$.
    calculated_spacing : list
        If a three item list (or similar mutable sequence) is given the
        members will be replaced with the actual calculated spacing in
        $A^{-1}$.

    Returns
    -------
    kpoint_grid : [int, int, int]
        MP grid specification to give the required spacing.

    """
    # No factor of 2pi in ase, everything in A^-1
    # reciprocal dimensions
    r_x, r_y, r_z = np.linalg.norm(atoms.get_reciprocal_cell(), axis=1)

    kpoint_grid = [int(r_x/spacing) + 1,
                   int(r_y/spacing) + 1,
                   int(r_z/spacing) + 1]

    if calculated_spacing is not None:
        calculated_spacing[:] = [r_x/kpoint_grid[0],
                                 r_y/kpoint_grid[1],
                                 r_z/kpoint_grid[2]]

    return kpoint_grid


def write_espresso_in(fd, atoms, input_data=None, pseudopotentials=None,
                      kspacing=None, kpts=None, koffset=(0, 0, 0),
                      **kwargs):
    """
    Create an input file for pw.x.

    Use set_initial_magnetic_moments to turn on spin, if ispin is set to 2
    with no magnetic moments, they will all be set to 0.0. Magnetic moments
    will be converted to the QE units (fraction of valence electrons) using
    any pseudopotential files found, or a best guess for the number of
    valence electrons.

    Units are not converted for any other input data, so use Quantum ESPRESSO
    units (Usually Ry or atomic units).

    Keys with a dimension (e.g. Hubbard_U(1)) will be incorporated as-is
    so the `i` should be made to match the output.

    Implemented features:

    - Conversion of :class:`ase.constraints.FixAtoms` and
                    :class:`ase.constraints.FixCartesian`.
    - `starting_magnetization` derived from the `mgmoms` and pseudopotentials
      (searches default paths for pseudo files.)
    - Automatic assignment of options to their correct sections.
    - Interpretation of ibrav (cell must exactly match the vectors defined
      in the QE docs).

    Not implemented:

    - Lists of k-points
    - Other constraints
    - Hubbard parameters
    - Validation of the argument types for input
    - Validation of required options
    - Reorientation for ibrav settings
    - Noncollinear magnetism

    Parameters
    ----------
    fd: file
        A file like object to write the input file to.
    atoms: Atoms
        A single atomistic configuration to write to `fd`.
    input_data: dict
        A flat or nested dictionary with input parameters for pw.x
    pseudopotentials: dict
        A filename for each atomic species, e.g.
        {'O': 'O.pbe-rrkjus.UPF', 'H': 'H.pbe-rrkjus.UPF'}.
        A dummy name will be used if none are given.
    kspacing: float
        Generate a grid of k-points with this as the minimum distance,
        in A^-1 between them in reciprocal space. If set to None, kpts
        will be used instead.
    kpts: (int, int, int) or dict
        If kpts is a tuple (or list) of 3 integers, it is interpreted
        as the dimensions of a Monkhorst-Pack grid.
        If kpts is a dict, it will either be interpreted as a path
        in the Brillouin zone (*) if it contains the 'path' keyword,
        otherwise it is converted to a Monkhorst-Pack grid (**).
        (*) see ase.dft.kpoints.bandpath
        (**) see ase.calculators.calculator.kpts2sizeandoffsets
    koffset: (int, int, int)
        Offset of kpoints in each direction. Must be 0 (no offset) or
        1 (half grid offset). Setting to True is equivalent to (1, 1, 1).

    """

    # Convert to a namelist to make working with parameters much easier
    # Note that the name ``input_data`` is chosen to prevent clash with
    # ``parameters`` in Calculator objects
    input_parameters = construct_namelist(input_data, **kwargs)

    # Convert ase constraints to QE constraints
    # Nx3 array of force multipliers matches what QE uses
    # Do this early so it is available when constructing the atoms card
    constraint_mask = np.ones((len(atoms), 3), dtype='int')
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            constraint_mask[constraint.index] = 0
        elif isinstance(constraint, FixCartesian):
            constraint_mask[constraint.a] = constraint.mask
        else:
            warnings.warn('Ignored unknown constraint {}'.format(constraint))

    # Deal with pseudopotentials
    # Look in all possible locations for the pseudos and try to figure
    # out the number of valence electrons
    pseudo_dirs = []
    if 'pseudo_dir' in input_parameters['control']:
        pseudo_dirs.append(input_parameters['control']['pseudo_dir'])
    if 'ESPRESSO_PSEUDO' in os.environ:
        pseudo_dirs.append(os.environ['ESPRESSO_PSEUDO'])
    pseudo_dirs.append(path.expanduser('~/espresso/pseudo/'))

    # Species info holds the information on the pseudopotential and
    # associated for each element
    if pseudopotentials is None:
        pseudopotentials = {}
    species_info = {}
    for species in set(atoms.get_chemical_symbols()):
        pseudo = pseudopotentials.get(species, '{}_dummy.UPF'.format(species))
        for pseudo_dir in pseudo_dirs:
            if path.exists(path.join(pseudo_dir, pseudo)):
                valence = grep_valence(path.join(pseudo_dir, pseudo))
                break
        else:  # not found in a file
            valence = SSSP_VALENCE[atomic_numbers[species]]

        species_info[species] = {'pseudo': pseudo,
                                 'valence': valence}

    # Convert atoms into species.
    # Each different magnetic moment needs to be a separate type even with
    # the same pseudopotential (e.g. an up and a down for AFM).
    # if any magmom are > 0 or nspin == 2 then use species labels.
    # Rememeber: magnetisation uses 1 based indexes
    atomic_species = OrderedDict()
    atomic_species_str = []
    atomic_positions_str = []

    nspin = input_parameters['system'].get('nspin', 1)  # 1 is the default
    if any(atoms.get_initial_magnetic_moments()):
        if nspin == 1:
            # Force spin on
            input_parameters['system']['nspin'] = 2
            nspin = 2

    if nspin == 2:
        # Spin on
        for atom, magmom in zip(atoms, atoms.get_initial_magnetic_moments()):
            if (atom.symbol, magmom) not in atomic_species:
                # spin as fraction of valence
                fspin = float(magmom) / species_info[atom.symbol]['valence']
                # Index in the atomic species list
                sidx = len(atomic_species) + 1
                # Index for that atom type; no index for first one
                tidx = sum(atom.symbol == x[0] for x in atomic_species) or ' '
                atomic_species[(atom.symbol, magmom)] = (sidx, tidx)
                # Add magnetization to the input file
                mag_str = 'starting_magnetization({0})'.format(sidx)
                input_parameters['system'][mag_str] = fspin
                atomic_species_str.append(
                    '{species}{tidx} {mass} {pseudo}\n'.format(
                        species=atom.symbol, tidx=tidx, mass=atom.mass,
                        pseudo=species_info[atom.symbol]['pseudo']))
            # lookup tidx to append to name
            sidx, tidx = atomic_species[(atom.symbol, magmom)]

            # only inclued mask if something is fixed
            if not all(constraint_mask[atom.index]):
                mask = ' {mask[0]} {mask[1]} {mask[2]}'.format(
                    mask=constraint_mask[atom.index])
            else:
                mask = ''

            # construct line for atomic positions
            atomic_positions_str.append(
                '{atom.symbol}{tidx} '
                '{atom.x:.10f} {atom.y:.10f} {atom.z:.10f}'
                '{mask}\n'.format(atom=atom, tidx=tidx, mask=mask))

    else:
        # Do nothing about magnetisation
        for atom in atoms:
            if atom.symbol not in atomic_species:
                atomic_species[atom.symbol] = True  # just a placeholder
                atomic_species_str.append(
                    '{species} {mass} {pseudo}\n'.format(
                        species=atom.symbol, mass=atom.mass,
                        pseudo=species_info[atom.symbol]['pseudo']))

            # only inclued mask if something is fixed
            if not all(constraint_mask[atom.index]):
                mask = ' {mask[0]} {mask[1]} {mask[2]}'.format(
                    mask=constraint_mask[atom.index])
            else:
                mask = ''

            atomic_positions_str.append(
                '{atom.symbol} '
                '{atom.x:.10f} {atom.y:.10f} {atom.z:.10f} '
                '{mask}\n'.format(atom=atom, mask=mask))

    # Add computed parameters
    # different magnetisms means different types
    input_parameters['system']['ntyp'] = len(atomic_species)
    input_parameters['system']['nat'] = len(atoms)

    # Use cell as given or fit to a specific ibrav
    if 'ibrav' in input_parameters['system']:
        ibrav = input_parameters['system']['ibrav']
        if ibrav != 0:
            celldm = cell_to_ibrav(atoms.cell, ibrav)
            regen_cell = ibrav_to_cell(celldm)[1]
            if not np.allclose(atoms.cell, regen_cell):
                warnings.warn('Input cell does not match requested ibrav'
                              '{} != {}'.format(regen_cell, atoms.cell))
            input_parameters['system'].update(celldm)
    else:
        # Just use standard cell block
        input_parameters['system']['ibrav'] = 0

    # Construct input file into this
    pwi = []

    # Assume sections are ordered (taken care of in namelist construction)
    # and that repr converts to a QE readable representation (except bools)
    for section in input_parameters:
        pwi.append('&{0}\n'.format(section.upper()))
        for key, value in input_parameters[section].items():
            if value is True:
                pwi.append('   {0:16} = .true.\n'.format(key))
            elif value is False:
                pwi.append('   {0:16} = .false.\n'.format(key))
            else:
                # repr format to get quotes around strings
                pwi.append('   {0:16} = {1!r:}\n'.format(key, value))
        pwi.append('/\n')  # terminate section
    pwi.append('\n')

    # Pseudopotentials
    pwi.append('ATOMIC_SPECIES\n')
    pwi.extend(atomic_species_str)
    pwi.append('\n')

    # KPOINTS - add a MP grid as required
    if kspacing is not None:
        kgrid = kspacing_to_grid(atoms, kspacing)
    elif kpts is not None:
        if isinstance(kpts, dict) and 'path' not in kpts:
            kgrid, shift = kpts2sizeandoffsets(atoms=atoms, **kpts)
            koffset = []
            for i, x in enumerate(shift):
                assert x == 0 or abs(x * kgrid[i] - 0.5) < 1e-14
                koffset.append(0 if x == 0 else 1)
        else:
            kgrid = kpts
    else:
        kgrid = (1, 1, 1)

    # True and False work here and will get converted by ':d' format
    if isinstance(koffset, int):
        koffset = (koffset, ) * 3

    if isinstance(kgrid, dict):
        pwi.append('K_POINTS crystal_b\n')
        assert 'path' in kgrid
        kgrid = kpts2ndarray(kgrid, atoms=atoms)
        pwi.append('%s\n' % len(kgrid))
        for k in kgrid:
            pwi.append('{k[0]:.14f} {k[1]:.14f} {k[2]:.14f} 0\n'.format(k=k))
        pwi.append('\n')
    elif all([x == 1 for x in kgrid]) and not any(koffset):
        # QE defaults to gamma point, make it explicit
        pwi.append('K_POINTS gamma\n')
        pwi.append('\n')
    else:
        pwi.append('K_POINTS automatic\n')
        pwi.append('{0[0]} {0[1]} {0[2]}  {1[0]:d} {1[1]:d} {1[2]:d}\n'
                   ''.format(kgrid, koffset))
        pwi.append('\n')

    # CELL block, if required
    if input_parameters['SYSTEM']['ibrav'] == 0:
        pwi.append('CELL_PARAMETERS angstrom\n')
        pwi.append('{cell[0][0]:.14f} {cell[0][1]:.14f} {cell[0][2]:.14f}\n'
                   '{cell[1][0]:.14f} {cell[1][1]:.14f} {cell[1][2]:.14f}\n'
                   '{cell[2][0]:.14f} {cell[2][1]:.14f} {cell[2][2]:.14f}\n'
                   ''.format(cell=atoms.cell))
        pwi.append('\n')

    # Positions - already constructed, but must appear after namelist
    pwi.append('ATOMIC_POSITIONS angstrom\n')
    pwi.extend(atomic_positions_str)
    pwi.append('\n')

    # DONE!
    fd.write(''.join(pwi))
