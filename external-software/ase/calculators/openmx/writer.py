"""
The ASE Calculator for OpenMX <http://www.openmx-square.org>: Python interface
to the software package for nano-scale material simulations based on density
functional theories.
    Copyright (C) 2018 JaeHwan Shim and JaeJun Yu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 2.1 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with ASE.  If not, see <http://www.gnu.org/licenses/>.
"""
import os
import numpy as np
from ase.units import Ha, Ry
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name
                                           , get_standard_key)
from ase.calculators.openmx import parameters as param

keys = [param.tuple_integer_keys, param.tuple_float_keys,
        param.tuple_bool_keys, param.integer_keys, param.float_keys,
        param.string_keys, param.bool_keys, param.list_int_keys,
        param.list_bool_keys, param.list_float_keys, param.matrix_keys]


def write_openmx(label=None, atoms=None, parameters=None, properties=None,
                 system_changes=None):
    """
    From atom image, 'images', write '.dat' file.
    First, set
    Write input (dat)-file.
    See calculator.py for further details.

    Parameters:
        - atoms        : The Atoms object to write.
        - properties   : The properties which should be calculated.
        - system_changes : List of properties changed since last run.
    """
    from ase.calculators.openmx import parameters as param
    filtered_keywords = parameters_to_keywords(label=label, atoms=atoms,
                                               parameters=parameters,
                                               properties=properties,
                                               system_changes=system_changes)
    keys = ['string', 'bool', 'integer', 'float',
            'tuple_integer', 'tuple_float', 'tuple_bool',
            'matrix', 'list_int', 'list_bool', 'list_float']
    # Start writing the file
    filename = get_file_name('.dat', label)
    with open(filename, 'w') as f:
        # Write 1-line keywords
        for fltrd_keyword in filtered_keywords.keys():
            for key in keys:
                openmx_keywords = getattr(param, key+'_keys')
                write = globals()['write_'+key]
                for omx_keyword in openmx_keywords:
                    if fltrd_keyword == get_standard_key(omx_keyword):
                        write(f, omx_keyword, filtered_keywords[fltrd_keyword])


def parameters_to_keywords(label=None, atoms=None, parameters=None,
                           properties=None, system_changes=None):
    """
    Before writing `label.dat` file, set up the ASE variables to OpenMX
    keywords. First, It initializes with given openmx keywords and reconstruct
    dictionary using standard parameters. If standard parameters and openmx
    keywords are contradict to each other, ignores openmx keyword.
     It includes,

    For asthetical purpose, sequnece of writing input file is specified.
    """
    from ase.calculators.openmx.parameters import matrix_keys
    from collections import OrderedDict
    keywords = OrderedDict()
    sequence = ['system_currentdirectory', 'system_name', 'data_path',
                'species_number', 'definition_of_atomic_species',
                'atoms_number', 'atoms_speciesandcoordinates_unit',
                'atoms_speciesandcoordinates', 'atoms_unitvectors_unit',
                'atoms_unitvectors', 'band_dispersion', 'band_nkpath',
                'band_kpath']

    for key in sequence:
        keywords[key] = None
    for key in parameters:
        if 'scf' in key:
            keywords[key] = None
    for key in parameters:
        if 'md' in key:
            keywords[key] = None

    # Initializes keywords to to given parameters
    for key in parameters.keys():
        keywords[key] = parameters[key]

    # Set up the single-line OpenMX keywords
    directory, prefix = os.path.split(label)
    curdir = os.path.join(os.getcwd(), prefix)
    keywords['system_currentdirectory'] = curdir  # Need absolute directory
    keywords['system_name'] = prefix
    keywords['data_path'] = os.environ.get('OPENMX_DFT_DATA_PATH')
    keywords['species_number'] = len(get_species(atoms.get_chemical_symbols()))
    keywords['atoms_number'] = len(atoms)
    keywords['atoms_unitvectors_unit'] = 'Ang'
    keywords['atoms_speciesandcoordinates_unit'] = 'Ang'
    keywords['scf_restart'] = parameters.get('scf_restart')
    if parameters.get('restart') is not None:
        keywords['scf_restart'] = True
    # Having generouse restart policy. It is dangerouse if one caluclate
    # totally different with previous calculator.

    if 'stress' in properties:
        keywords['scf_stress_tensor'] = True

    # keywords['scf_stress_tensor'] = 'stress' in properties
    # This is not working due to the UnitCellFilter method.

    # Set up standard parameters to openmx keyword
    keywords['scf_maxiter'] = parameters.get('maxiter')
    keywords['scf_xctype'] = get_xc(parameters.get('xc'))
    keywords['scf_energycutoff'] = parameters.get('energy_cutoff') / Ry
    keywords['scf_criterion'] = parameters.get('convergence') / Ha
    keywords['scf_kgrid'] = get_scf_kgrid(
                                        kpts=parameters.get('kpts'),
                                        scf_kgrid=parameters.get('scf_kgrid'),
                                        atoms=atoms)
    keywords['scf_eigenvaluesolver'] = get_eigensolver(atoms, parameters)
    keywords['scf_spinpolarization'] = get_spinpol(atoms, parameters)
    keywords['scf_external_fields'] = parameters.get('external')
    keywords['scf_mixing_type'] = parameters.get('mixer')
    keywords['scf_electronic_temperature'] = parameters.get('smearing')
    keywords['scf_system_charge'] = parameters.get('charge')
    if parameters.get('band_kpath') is not None:
        keywords['band_dispersion'] = True
    keywords['band_nkpath'] = parameters.get('band_kpath')
    if keywords['band_nkpath'] is not None:
        keywords['band_nkpath'] = len(keywords['band_nkpath'])

    # Set up Wannier Environment
    if parameters.get('wannier_func_calc') is not None:
        keywords['species_number'] *= 2

    # Set up the matrix-type OpenMX keywords
    for key in matrix_keys:
        get_matrix_key = globals()['get_'+get_standard_key(key)]
        keywords[get_standard_key(key)] = get_matrix_key(atoms, parameters)
    return OrderedDict([(k, v)for k, v in keywords.items()
                        if not(v is None or
                               (isinstance(v, list) and v == []))])


def get_species(symbols):
    species = []
    [species.append(s) for s in symbols if s not in species]
    return species


def get_xc(xc):
    if xc in ['PBE', 'GGA', 'GGA-PBE']:
        return 'GGA-PBE'
    elif xc in ['LDA']:
        return 'LDA'
    elif xc in ['CA', 'PW']:
        return 'LSDA-' + xc
    elif xc in ['LSDA']:
        return 'LSDA-CA'
    else:
        return 'LDA'


def get_eigensolver(atoms, parameters):
    if get_atoms_unitvectors(atoms, parameters) is None:
        return 'Cluster'
    else:
        eigensolver = parameters.get('scf_eigenvaluesolver', 'Band')
        return parameters.get('eigensolver', eigensolver)


def get_scf_kgrid(kpts=None, scf_kgrid=None, atoms=None):
    if isinstance(kpts, tuple) or isinstance(kpts, list):
        if len(kpts) == 3 and isinstance(kpts[0], int):
            return kpts
        elif scf_kgrid is not None:
            return scf_kgrid
        else:
            return (4, 4, 4)
    elif isinstance(kpts, float) or isinstance(kpts, int):
        return tuple(kpts2sizeandoffsets(atoms=atoms, density=kpts)[0])
    else:
        return (4, 4, 4)


def get_definition_of_atomic_species(atoms, parameters):
    """
    Using atoms and parameters, Returns the list `definition_of_atomic_species`
    where matrix of strings contains the information between keywords.
    For example,
     definition_of_atomic_species =
         [['H','H5.0-s1>1p1>1','H_CA13'],
          ['C','C5.0-s1>1p1>1','C_CA13']]
    Goes to,
      <Definition.of.Atomic.Species
        H   H5.0-s1>1p1>1      H_CA13
        C   C5.0-s1>1p1>1      C_CA13
      Definition.of.Atomic.Species>
    Further more, you can specify the wannier infomation here.
    A. Define local functions for projectors
      Since the pseudo-atomic orbitals are used for projectors,
      the specification of them is the same as for the basis functions.
      An example setting, for silicon in diamond structure, is as following:
   Species.Number          2
      <Definition.of.Atomic.Species
        Si       Si7.0-s2p2d1    Si_CA13
        proj1    Si5.5-s1p1d1f1  Si_CA13
      Definition.of.Atomic.Species>
    """
    if parameters.get('definition_of_atomic_species') is not None:
        return parameters['definition_of_atomic_species']
    definition_of_atomic_species = []
    xc = parameters.get('scf_xctype')
    xc = parameters.get('xc')
    chem = atoms.get_chemical_symbols()
    species = get_species(chem)
    for element in species:
        rad_orb = get_cutoff_radius_and_orbital(element=element)
        potential = get_pseudo_potential_suffix(element=element, xc=xc)
        definition_of_atomic_species.append([element, rad_orb, potential])
    # Put the same orbital and radii with chemical symbol.
    wannier_projectors = parameters.get('definition_of_wannier_projectors', [])
    for i, projector in enumerate(wannier_projectors):
        full_projector = definition_of_atomic_species[i]
        full_projector[0] = projector
        definition_of_atomic_species.append(full_projector)
    return definition_of_atomic_species


def get_cutoff_radius_and_orbital(element=None, orbital=None):
    """
    For a given element, retruns the string specifying cutoff radius and
    orbital using default_settings.py. For example,
       'Si'   ->   'Si.7.0-s2p2d1'
    If one wannts to change the atomic radius for a special purpose, one should
    change the default_settings.py directly.
    """
    from ase.calculators.openmx import default_settings
    orbital = element
    orbital_letters = ['s', 'p', 'd', 'f', 'g', 'h']
    default_dictionary = default_settings.default_dictionary
    orbital_numbers = default_dictionary[element]['orbitals used']
    cutoff_radius = default_dictionary[element]['cutoff radius']
    orbital += "%.1f" % float(cutoff_radius) + '-'
    for i, orbital_number in enumerate(orbital_numbers):
        orbital += orbital_letters[i] + str(orbital_number)
    return orbital


def get_pseudo_potential_suffix(element=None, xc=None):
    """
    For a given element, returns the string specifying pseudo potential suffix.
    For example,
        'Si'   ->   'Si_CA13'
    We used 2013 version of pseudo potential
    """
    from ase.calculators.openmx import default_settings
    default_dictionary = default_settings.default_dictionary
    pseudo_potential_suffix = element
    xc_label = {'PBE': 'PBE', 'GGA': 'PBE', 'GGA-PBE': 'PBE'}
    suffix = default_dictionary[element]['pseudo-potential suffix']
    pseudo_potential_suffix += '_' + xc_label.get(xc, 'CA') + suffix + '13'
    return pseudo_potential_suffix


def get_atoms_speciesandcoordinates(atoms, parameters):
    """
    The atomic coordinates and the number of spin charge are given by the
    keyword
    'Atoms.SpeciesAndCoordinates' as follows:
    <Atoms.SpeciesAndCoordinates
     1  Mn    0.00000   0.00000   0.00000   8.0  5.0  45.0 0.0 45.0 0.0  1 on
     2  O     1.70000   0.00000   0.00000   3.0  3.0  45.0 0.0 45.0 0.0  1 on
    Atoms.SpeciesAndCoordinates>
    to know more, link <http://www.openmx-square.org/openmx_man3.7/node85.html>
    """
    atoms_speciesandcoordinates = []
    xc = parameters.get('xc')
    # Appending number and elemental symbol
    elements = atoms.get_chemical_symbols()
    for i, element in enumerate(elements):
        atoms_speciesandcoordinates.append([str(i+1), element])
    # Appending positions
    positions = atoms.get_positions()
    for i, position in enumerate(positions):
        atoms_speciesandcoordinates[i].extend(position)
    # Appending magnetic moment
    magmoms = atoms.get_initial_magnetic_moments()
    for i, magmom in enumerate(magmoms):
        up_down_spin = get_up_down_spin(magmom, elements[i], xc)
        atoms_speciesandcoordinates[i].extend(up_down_spin)
    # Appending magnetic field Spin magnetic moment theta phi
    spin_directions = get_spin_direction(magmoms)
    for i, spin_direction in enumerate(spin_directions):
        atoms_speciesandcoordinates[i].extend(spin_direction)
    # Appending magnetic field for Orbital magnetic moment theta phi
    orbital_directions = get_orbital_direction()
    for i, orbital_direction in enumerate(orbital_directions):
        atoms_speciesandcoordinates[i].extend(orbital_direction)
    # Appending Noncolinear schem switch
    noncollinear_switches = get_noncollinear_switches()
    for i, noncollinear_switch in enumerate(noncollinear_switches):
        atoms_speciesandcoordinates[i].extend(noncollinear_switch)
    # Appending orbital_enhancement_switch
    lda_u_switches = get_lda_u_switches()
    for i, lda_u_switch in enumerate(lda_u_switches):
        atoms_speciesandcoordinates[i].extend(lda_u_switch)
    return atoms_speciesandcoordinates


def get_up_down_spin(magmom, element, xc):
    magmom = np.linalg.norm(magmom)
    filename = get_pseudo_potential_suffix(element, xc)
    valence_electron = float(read_electron_valency(filename))
    return [valence_electron/2+magmom/2, valence_electron/2-magmom/2]


def get_spin_direction(magmoms):
    '''
    From atoms.magmom, returns the spin direction of phi and theta
    '''
    if np.array(magmoms).dtype == float or \
       np.array(magmoms).dtype is np.float64:
        return []
    else:
        magmoms = np.array(magmoms)
        return magmoms/np.linalg.norm(magmoms, axis=1)


def get_orbital_direction():
    orbital_direction = []
    # print("Not Implemented Yet")
    return orbital_direction


def get_noncollinear_switches():
    noncolinear_switches = []
    # print("Not Implemented Yet")
    return noncolinear_switches


def get_lda_u_switches():
    lda_u_switches = []
    # print("Not Implemented Yet")
    return lda_u_switches


def get_spinpol(atoms, parameters):
    ''' Judgeds the keyword 'scf.SpinPolarization'
     If the keyword is not None, spinpol gets the keyword by following priority
       1. standard_spinpol
       2. scf_spinpolarization
       3. magnetic moments of atoms
    '''
    standard_spinpol = parameters.get('spinpol', None)
    scf_spinpolarization = parameters.get('scf_spinpolarization', None)
    m = atoms.get_initial_magnetic_moments()
    syn = {True: 'On', False: None, 'on': 'On', 'off': None,
           None: None, 'nc': 'NC'}
    spinpol = np.any(m >= 0.1)
    if scf_spinpolarization is not None:
        spinpol = scf_spinpolarization
    if standard_spinpol is not None:
        spinpol = standard_spinpol
    if isinstance(spinpol, str):
        spinpol = spinpol.lower()
    return syn[spinpol]


def get_atoms_unitvectors(atoms, parameters):
    zero_vec = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    if np.all(atoms.get_cell() == zero_vec) is True:
        default_cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return parameters.get('atoms_unitvectors', default_cell)
    atoms_unitvectors = atoms.get_cell().T
    return atoms_unitvectors


def get_hubbard_u_values(atoms, parameters):
    return parameters.get('hubbard_u_values', [])


def get_atoms_cont_orbitals(atoms, parameters):
    return parameters.get('atoms_cont_orbitals', [])


def get_md_fixed_xyz(atoms, parameters):
    return parameters.get('md_fixed_xyz', [])


def get_md_tempcontrol(atoms, parameters):
    return parameters.get('md_tempcontrol', [])


def get_md_init_velocity(atoms, parameters):
    return parameters.get('md_init_velocity', [])


def get_band_kpath_unitcell(atoms, parameters):
    return parameters.get('band_kpath_unitcell', [])


def get_band_kpath(atoms, parameters):
    kpts = parameters.get('kpts')
    if isinstance(kpts, list) and len(kpts) > 3:
        return get_kpath(kpts=kpts)
    else:
        return parameters.get('band_kpath', [])


def get_mo_kpoint(atoms, parameters):
    return parameters.get('get_mo_kpoint', [])


def get_wannier_initial_projectors(atoms, parameters):
    """
    B. Specify the orbital, central position and orientation of a projector
    Wannier.Initial.Projectos will be used to specify the projector name,
    local orbital function, center of local orbital, and the local z-axis and
    x-axis for orbital orientation.

    An example setting is shown here:
    wannier_initial_projectors=
    [['proj1-sp3','0.250','0.250','0.25','-1.0','0.0','0.0','0.0','0.0','-1.0']
    ,['proj1-sp3','0.000','0.000','0.00','0.0','0.0','1.0','1.0','0.0','0.0']]
    Goes to,
        <Wannier.Initial.Projectors
           proj1-sp3   0.250  0.250  0.250   -1.0 0.0 0.0    0.0  0.0 -1.0
           proj1-sp3   0.000  0.000  0.000    0.0 0.0 1.0    1.0  0.0  0.0
        Wannier.Initial.Projectors>
    """
    return parameters.get('wannier_initial_projectors', [])


def get_kpath(self, kpts=None, symbols=None, band_kpath=None, eps=1e-5):
        """
        Convert band_kpath <-> kpts. Symbols will be guess automatically
        by using dft space group method
        For example,
        kpts  = [(0, 0, 0), (0.125, 0, 0) ... (0.875, 0, 0),
                 (1, 0, 0), (1, 0.0625, 0) .. (1, 0.4375,0),
                 (1, 0.5,0),(0.9375, 0.5,0).. (    ...    ),
                 (0.5, 0.5, 0.5) ...               ...     ,
                    ...          ...               ...     ,
                    ...        (0.875, 0, 0),(1.0, 0.0, 0.0)]
        band_kpath =
        [['15','0.0','0.0','0.0','1.0','0.0','0.0','g','X'],
         ['15','1.0','0.0','0.0','1.0','0.5','0.0','X','W'],
         ['15','1.0','0.5','0.0','0.5','0.5','0.5','W','L'],
         ['15','0.5','0.5','0.5','0.0','0.0','0.0','L','g'],
         ['15','0.0','0.0','0.0','1.0','0.0','0.0','g','X']]
        where, it will be written as
         <Band.kpath
          15  0.0 0.0 0.0   1.0 0.0 0.0   g X
          15  1.0 0.0 0.0   1.0 0.5 0.0   X W
          15  1.0 0.5 0.0   0.5 0.5 0.5   W L
          15  0.5 0.5 0.5   0.0 0.0 0.0   L g
          15  0.0 0.0 0.0   1.0 0.0 0.0   g X
         Band.kpath>
        """
        if kpts is None:
            kx_linspace = np.linspace(band_kpath[0]['start_point'][0],
                                      band_kpath[0]['end_point'][0],
                                      band_kpath[0][0])
            ky_linspace = np.linspace(band_kpath[0]['start_point'][1],
                                      band_kpath[0]['end_point'][1],
                                      band_kpath[0]['kpts'])
            kz_linspace = np.linspace(band_kpath[0]['start_point'][2],
                                      band_kpath[0]['end_point'][2],
                                      band_kpath[0]['kpts'])
            kpts = np.array([kx_linspace, ky_linspace, kz_linspace]).T
            for path in band_kpath[1:]:
                kx_linspace = np.linspace(path['start_point'][0],
                                          path['end_point'][0],
                                          path['kpts'])
                ky_linspace = np.linspace(path['start_point'][1],
                                          path['end_point'][1],
                                          path['kpts'])
                kz_linspace = np.linspace(path['start_point'][2],
                                          path['end_point'][2],
                                          path['kpts'])
                k_lin = np.array([kx_linspace, ky_linspace, kz_linspace]).T
                kpts = np.append(kpts, k_lin, axis=0)
            return kpts
        elif band_kpath is None:
            band_kpath = []
            points = np.asarray(kpts)
            diffs = points[1:] - points[:-1]
            kinks = abs(diffs[1:] - diffs[:-1]).sum(1) > eps
            N = len(points)
            indices = [0]
            indices.extend(np.arange(1, N - 1)[kinks])
            indices.append(N - 1)
            for start, end, s_sym, e_sym in zip(indices[1:], indices[:-1],
                                                symbols[1:], symbols[:-1]):
                band_kpath.append({'start_point': start, 'end_point': end,
                                   'kpts': 20,
                                   'path_symbols': (s_sym, e_sym)})
        else:
            raise KeyError('You should specify band_kpath or kpts')
            return band_kpath


def write_string(f, key, value):
    f.write("        ".join([key, value]))
    f.write("\n")


def write_tuple_integer(f, key, value):
    f.write("        ".join([key, "%d %d %d" % value]))
    f.write("\n")


def write_tuple_float(f, key, value):
    f.write("        ".join([key, "%.4f %.4f %.4f" % value]))
    f.write("\n")


def write_tuple_bool(f, key, value):
    omx_bl = {True: 'On', False: 'Off'}
    f.write("        ".join([key, "%s %s %s" % [omx_bl[bl] for bl in value]]))
    f.write("\n")


def write_integer(f, key, value):
    f.write("        ".join([key, "%d" % value]))
    f.write("\n")


def write_float(f, key, value):
    f.write("        ".join([key, "%.8g" % value]))
    f.write("\n")


def write_bool(f, key, value):
    omx_bl = {True: 'On', False: 'Off'}
    f.write("        ".join([key, "%s" % omx_bl[value]]))
    f.write("\n")


def write_list_int(f, key, value):
    f.write("".join(key) + "     ".join(map(str, value)))


def write_list_bool(f, key, value):
    omx_bl = {True: 'On', False: 'Off'}
    f.write("".join(key) + "     ".join([omx_bl[bl] for bl in value]))


def write_list_float(f, key, value):
    f.write("".join(key) + "     ".join(map(str, value)))


def write_matrix(f, key, value):
    f.write('<' + key)
    f.write("\n")
    for line in value:
        f.write("    "+"  ".join(map(str, line)))
        f.write("\n")
    f.write(key + '>')
    f.write("\n\n")


def get_openmx_key(key):
    """
    For the writting purpose, we need to know Original OpenMX keyword format.
    By comparing keys in the parameters.py, restore the original key
    """
    for openmx_key in keys:
        for openmx_keyword in openmx_key:
            if key == get_standard_key(openmx_keyword):
                return openmx_keyword
