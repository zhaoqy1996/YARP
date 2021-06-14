"""
This module defines the ASE interface to SIESTA.

Written by Mads Engelund
http://www.mads-engelund.net

Home of the SIESTA package:
http://www.uam.es/departamentos/ciencias/fismateriac/siesta

2017.04 - Pedro Brandimarte: changes for python 2-3 compatible

"""

from __future__ import print_function
import os
from os.path import join, isfile, islink
import numpy as np
import shutil
from ase.units import Ry, eV, Bohr
from ase.data import atomic_numbers
from ase.calculators.siesta.import_functions import read_rho, xv_to_atoms
from ase.calculators.siesta.import_functions import \
    get_valence_charge, read_vca_synth_block
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.siesta.parameters import PAOBasisBlock, Species
from ase.calculators.siesta.parameters import format_fdf

meV = 0.001 * eV


class SiestaParameters(Parameters):
    """Parameters class for the calculator.
    Documented in BaseSiesta.__init__

    """
    def __init__(
            self,
            label='siesta',
            mesh_cutoff=200 * Ry,
            energy_shift=100 * meV,
            kpts=None,
            xc='LDA',
            basis_set='DZP',
            spin='UNPOLARIZED',
            species=tuple(),
            pseudo_qualifier=None,
            pseudo_path=None,
            atoms=None,
            restart=None,
            ignore_bad_restart_file=False,
            fdf_arguments=None):
        kwargs = locals()
        kwargs.pop('self')
        Parameters.__init__(self, **kwargs)


class BaseSiesta(FileIOCalculator):
    """Calculator interface to the SIESTA code.
    """
    allowed_basis_names = ['SZ', 'SZP', 'DZ', 'DZP']
    allowed_spins = ['UNPOLARIZED', 'COLLINEAR', 'FULL']
    allowed_xc = {}
    allowed_fdf_keywords = {}
    unit_fdf_keywords = {}
    implemented_properties = (
        'energy',
        'forces',
        'stress',
        'dipole',
        'eigenvalues',
        'density',
        'fermi_energy')

    # Dictionary of valid input vaiables.
    default_parameters = SiestaParameters()

    def __init__(self, **kwargs):
        """ASE interface to the SIESTA code.

        Parameters:
           - label        : The base head of all created files.
           - mesh_cutoff  : Energy in eV.
                            The mesh cutoff energy for determining number of
                            grid points.
           - energy_shift : Energy in eVV
                            The confining energy of the basis sets.
           - kpts         : Tuple of 3 integers, the k-points in different
                            directions.
           - xc           : The exchange-correlation potential. Can be set to
                            any allowed value for either the Siesta
                            XC.funtional or XC.authors keyword. Default "LDA"
           - basis_set    : "SZ"|"SZP"|"DZ"|"DZP", strings which specify the
                            type of functions basis set.
           - spin         : "UNPOLARIZED"|"COLLINEAR"|"FULL". The level of spin
                            description to be used.
           - species      : None|list of Species objects. The species objects
                            can be used to to specify the basis set,
                            pseudopotential and whether the species is ghost.
                            The tag on the atoms object and the element is used
                            together to identify the species.
           - pseudo_path  : None|path. This path is where
                            pseudopotentials are taken from.
                            If None is given, then then the path given
                            in $SIESTA_PP_PATH will be used.
           - pseudo_qualifier: None|string. This string will be added to the
                            pseudopotential path that will be retrieved.
                            For hydrogen with qualifier "abc" the
                            pseudopotential "H.abc.psf" will be retrieved.
           - atoms        : The Atoms object.
           - restart      : str.  Prefix for restart file.
                            May contain a directory.
                            Default is  None, don't restart.
           - siesta_default: Use siesta default parameter if the parameter
                            is not explicitly set.
           - ignore_bad_restart_file: bool.
                            Ignore broken or missing restart file.
                            By default, it is an error if the restart
                            file is missing or broken.
           - fdf_arguments: Explicitly given fdf arguments. Dictonary using
                            Siesta keywords as given in the manual. List values
                            are written as fdf blocks with each element on a
                            separate line, while tuples will write each element
                            in a single line.  ASE units are assumed in the
                            input.
        """

        # Put in the default arguments.
        parameters = self.default_parameters.__class__(**kwargs)

        # Setup the siesta command based on number of nodes.
        command = os.environ.get('SIESTA_COMMAND')
        if command is None:
            mess = "The 'SIESTA_COMMAND' environment is not defined."
            raise ValueError(mess)

        label = parameters['label']
        self.label = label

        runfile = label + '.fdf'
        outfile = label + '.out'
        try:
            command = command % (runfile, outfile)
        except TypeError:
            raise ValueError(
                "The 'SIESTA_COMMAND' environment must " +
                "be a format string" +
                " with two string arguments.\n" +
                "Example : 'siesta < ./%s > ./%s'.\n" +
                "Got '%s'" % command)

        # Call the base class.
        FileIOCalculator.__init__(
            self,
            command=command,
            **parameters)

    def __getitem__(self, key):
        """Convenience method to retrieve a parameter as
        calculator[key] rather than calculator.parameters[key]

            Parameters:
                -key       : str, the name of the parameters to get.
        """
        return self.parameters[key]

    def species(self, atoms):
        """Find all relevant species depending on the atoms object and
        species input.

            Parameters :
                - atoms : An Atoms object.
        """
        # For each element use default species from the species input, or set
        # up a default species  from the general default parameters.
        symbols = np.array(atoms.get_chemical_symbols())
        tags = atoms.get_tags()
        species = list(self['species'])
        default_species = [
            s for s in species
            if (s['tag'] is None) and s['symbol'] in symbols]
        default_symbols = [s['symbol'] for s in default_species]
        for symbol in symbols:
            if symbol not in default_symbols:
                spec = Species(symbol=symbol,
                               basis_set=self['basis_set'],
                               tag=None)
                default_species.append(spec)
                default_symbols.append(symbol)
        assert len(default_species) == len(np.unique(symbols))

        # Set default species as the first species.
        species_numbers = np.zeros(len(atoms), int)
        i = 1
        for spec in default_species:
            mask = symbols == spec['symbol']
            species_numbers[mask] = i
            i += 1

        # Set up the non-default species.
        non_default_species = [s for s in species if not s['tag'] is None]
        for spec in non_default_species:
            mask1 = (tags == spec['tag'])
            mask2 = (symbols == spec['symbol'])
            mask = np.logical_and(mask1, mask2)
            if sum(mask) > 0:
                species_numbers[mask] = i
                i += 1
        all_species = default_species + non_default_species

        return all_species, species_numbers

    def set(self, **kwargs):
        """Set all parameters.

            Parameters:
                -kwargs  : Dictionary containing the keywords defined in
                           SiestaParameters.
        """
        # Find not allowed keys.
        default_keys = list(self.__class__.default_parameters)
        offending_keys = set(kwargs) - set(default_keys)
        if len(offending_keys) > 0:
            mess = "'set' does not take the keywords: %s "
            raise ValueError(mess % list(offending_keys))

        # Check energy inputs.
        for arg in ['mesh_cutoff', 'energy_shift']:
            value = kwargs.get(arg)
            if value is None:
                continue
            if not (isinstance(value, (float, int)) and value > 0):
                mess = "'%s' must be a positive number(in eV), \
                    got '%s'" % (arg, value)
                raise ValueError(mess)

        # Check the basis set input.
        if 'basis_set' in kwargs:
            basis_set = kwargs['basis_set']
            allowed = self.allowed_basis_names
            if not (isinstance(basis_set, PAOBasisBlock) or
                    basis_set in allowed):
                mess = "Basis must be either %s, got %s" % (allowed, basis_set)
                raise ValueError(mess)

        # Check the spin input.
        if 'spin' in kwargs:
            spin = kwargs['spin']
            if spin is not None and (spin not in self.allowed_spins):
                mess = "Spin must be %s, got %s" % (self.allowed_spins, spin)
                raise ValueError(mess)

        # Check the functional input.
        xc = kwargs.get('xc')
        if isinstance(xc, (tuple, list)) and len(xc) == 2:
            functional, authors = xc
            if functional not in self.allowed_xc:
                mess = "Unrecognized functional keyword: '%s'" % functional
                raise ValueError(mess)
            if authors not in self.allowed_xc[functional]:
                mess = "Unrecognized authors keyword for %s: '%s'"
                raise ValueError(mess % (functional, authors))

        elif xc in self.allowed_xc:
            functional = xc
            authors = self.allowed_xc[xc][0]
        else:
            found = False
            for key, value in self.allowed_xc.items():
                if xc in value:
                    found = True
                    functional = key
                    authors = xc
                    break

            if not found:
                raise ValueError("Unrecognized 'xc' keyword: '%s'" % xc)
        kwargs['xc'] = (functional, authors)

        # Check fdf_arguments.
        fdf_arguments = kwargs.get('fdf_arguments')
        self.validate_fdf_arguments(fdf_arguments)

        FileIOCalculator.set(self, **kwargs)

    def set_fdf_arguments(self, fdf_arguments):
        """ Set the fdf_arguments after the initialization of the
            calculator.
        """
        self.validate_fdf_arguments(fdf_arguments)
        FileIOCalculator.set(self, fdf_arguments=fdf_arguments)

    def validate_fdf_arguments(self, fdf_arguments):
        """ Raises error if the fdf_argument input is not a
            dictionary of allowed keys.
        """
        # None is valid
        if fdf_arguments is None:
            return

        # Type checking.
        if not isinstance(fdf_arguments, dict):
            raise TypeError("fdf_arguments must be a dictionary.")

        # Check if keywords are allowed.
        fdf_keys = set(fdf_arguments)
        allowed_keys = set(self.allowed_fdf_keywords)
        if not fdf_keys.issubset(allowed_keys):
            offending_keys = fdf_keys.difference(allowed_keys)
            raise ValueError("The 'fdf_arguments' dictionary " +
                             "argument does not allow " +
                             "the keywords: %s" % str(offending_keys))

    def calculate(self,
                  atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        """Capture the RuntimeError from FileIOCalculator.calculate
        and add a little debug information from the Siesta output.

        See base FileIocalculator for documentation.
        """

        try:
            FileIOCalculator.calculate(
                self,
                atoms=atoms,
                properties=properties,
                system_changes=system_changes)

        # Here a test to check if the potential are in the right place!!!
        except RuntimeError as e:
            try:
                with open(self.label + '.out', 'r') as f:
                    lines = f.readlines()
                debug_lines = 10
                print('##### %d last lines of the Siesta output' % debug_lines)
                for line in lines[-20:]:
                    print(line.strip())
                print('##### end of siesta output')
                raise e
            except:
                raise e

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write input (fdf)-file.
        See calculator.py for further details.

        Parameters:
            - atoms        : The Atoms object to write.
            - properties   : The properties which should be calculated.
            - system_changes : List of properties changed since last run.
        """
        # Call base calculator.
        FileIOCalculator.write_input(
            self,
            atoms=atoms,
            properties=properties,
            system_changes=system_changes)

        if system_changes is None and properties is None:
            return

        filename = self.label + '.fdf'

        # On any changes, remove all analysis files.
        if system_changes is not None:
            self.remove_analysis()

        # Start writing the file.
        with open(filename, 'w') as f:
            # Write system name and label.
            f.write(format_fdf('SystemName', self.label))
            f.write(format_fdf('SystemLabel', self.label))
            f.write("\n")

            # Write the minimal arg
            self._write_species(f, atoms)
            self._write_structure(f, atoms)

            # First write explicitly given options to
            # allow the user to overwrite anything.
            self._write_fdf_arguments(f)

            # Use the saved density matrix if only 'cell' and 'positions'
            # haved changes.
            if (system_changes is None or
                ('numbers' not in system_changes and
                 'initial_magmoms' not in system_changes and
                 'initial_charges' not in system_changes)):
                f.write(format_fdf('DM.UseSaveDM', True))

            # Save density.
            if 'density' in properties:
                f.write(format_fdf('SaveRho', True))

            # Force siesta to return error on no convergence.
            # Why?? maybe we don't want to force convergency??
            # f.write(format_fdf('SCFMustConverge', True))

            self._write_kpts(f)

    def read(self, filename):
        """Read parameters from file."""
        if not os.path.exists(filename):
            raise ReadError("The restart file '%s' does not exist" % filename)
        self.atoms = xv_to_atoms(filename)
        self.read_results()

    def _write_fdf_arguments(self, f):
        """Write directly given fdf-arguments.
        """
        fdf_arguments = self.parameters['fdf_arguments']
        if fdf_arguments is None:
            fdf_arguments = {}
        fdf_arguments["XC.functional"], \
            fdf_arguments["XC.authors"] = self.parameters['xc']
        energy_shift = self['energy_shift']
        fdf_arguments["PAO.EnergyShift"] = energy_shift
        mesh_cutoff = '%.4f eV' % self['mesh_cutoff']
        fdf_arguments["MeshCutoff"] = mesh_cutoff
        if self['spin'] == 'UNPOLARIZED':
            fdf_arguments["SpinPolarized"] = False
        elif self['spin'] == 'COLLINEAR':
            fdf_arguments["SpinPolarized"] = True
        elif self['spin'] == 'FULL':
            fdf_arguments["SpinPolarized"] = True
            fdf_arguments["NonCollinearSpin"] = True

        for key, value in self.allowed_fdf_keywords.items():
            if key in fdf_arguments.keys():
                if key in self.unit_fdf_keywords:
                    val = '%.8f %s' % (fdf_arguments[key],
                                       self.unit_fdf_keywords[key])
                    f.write(format_fdf(key, val))
                elif fdf_arguments[key] != value:
                    f.write(format_fdf(key, fdf_arguments[key]))

    def remove_analysis(self):
        """ Remove all analysis files"""
        filename = self.label + '.RHO'
        if os.path.exists(filename):
            os.remove(filename)

    def _write_structure(self, f, atoms):
        """Translate the Atoms object to fdf-format.

        Parameters:
            - f:     An open file object.
            - atoms: An atoms object.
        """
        unit_cell = atoms.get_cell()
        f.write('\n')

        # Write lattice vectors
        if np.any(unit_cell):
            f.write(format_fdf('LatticeConstant', '1.0 Ang'))
            f.write('%block LatticeVectors\n')
            for i in range(3):
                for j in range(3):
                    s = ('    %.15f' % unit_cell[i, j]).rjust(16) + ' '
                    f.write(s)
                f.write('\n')
            f.write('%endblock LatticeVectors\n')
            f.write('\n')

        self._write_atomic_coordinates(f, atoms)

        # Write magnetic moments.
        magmoms = atoms.get_initial_magnetic_moments()

        # The DM.InitSpin block must be written to initialize to
        # no spin. SIESTA default is FM initialization, if the
        # block is not written, but  we must conform to the
        # atoms object.
        if self['spin'] != 'UNPOLARIZED':
            f.write('%block DM.InitSpin\n')
            for n, M in enumerate(magmoms):
                if M != 0:
                    f.write('    %d %.14f\n' % (n + 1, M))
            f.write('%endblock DM.InitSpin\n')
            f.write('\n')

    def _write_atomic_coordinates(self, f, atoms):
        """Write atomic coordinates.

        Parameters:
            - f:     An open file object.
            - atoms: An atoms object.
        """
        species, species_numbers = self.species(atoms)
        f.write('\n')
        f.write('AtomicCoordinatesFormat  Ang\n')
        f.write('%block AtomicCoordinatesAndAtomicSpecies\n')
        for atom, number in zip(atoms, species_numbers):
            xyz = atom.position
            line = ('    %.9f' % xyz[0]).rjust(16) + ' '
            line += ('    %.9f' % xyz[1]).rjust(16) + ' '
            line += ('    %.9f' % xyz[2]).rjust(16) + ' '
            line += str(number) + '\n'
            f.write(line)
        f.write('%endblock AtomicCoordinatesAndAtomicSpecies\n')
        f.write('\n')

        origin = tuple(-atoms.get_celldisp().flatten())
        if any(origin):
            f.write('%block AtomicCoordinatesOrigin\n')
            f.write('     %.4f  %.4f  %.4f\n' % origin)
            f.write('%endblock AtomicCoordinatesOrigin\n')
            f.write('\n')

    def _write_kpts(self, f):
        """Write kpts.

        Parameters:
            - f : Open filename.
        """
        if self["kpts"] is None:
            return
        kpts = np.array(self['kpts'])
        f.write('\n')
        f.write('#KPoint grid\n')
        f.write('%block kgrid_Monkhorst_Pack\n')

        for i in range(3):
            s = ''
            if i < len(kpts):
                number = kpts[i]
                displace = 0.0
            else:
                number = 1
                displace = 0
            for j in range(3):
                if j == i:
                    write_this = number
                else:
                    write_this = 0
                s += '     %d  ' % write_this
            s += '%1.1f\n' % displace
            f.write(s)
        f.write('%endblock kgrid_Monkhorst_Pack\n')
        f.write('\n')

    def _write_species(self, f, atoms):
        """Write input related the different species.

        Parameters:
            - f:     An open file object.
            - atoms: An atoms object.
        """
        species, species_numbers = self.species(atoms)

        if not self['pseudo_path'] is None:
            pseudo_path = self['pseudo_path']
        elif 'SIESTA_PP_PATH' in os.environ:
            pseudo_path = os.environ['SIESTA_PP_PATH']
        else:
            mess = "Please set the environment variable 'SIESTA_PP_PATH'"
            raise Exception(mess)

        f.write(format_fdf('NumberOfSpecies', len(species)))
        f.write(format_fdf('NumberOfAtoms', len(atoms)))

        pao_basis = []
        chemical_labels = []
        basis_sizes = []
        synth_blocks = []
        for species_number, spec in enumerate(species):
            species_number += 1
            symbol = spec['symbol']
            atomic_number = atomic_numbers[symbol]

            if spec['pseudopotential'] is None:
                if self.pseudo_qualifier() == '':
                    label = symbol
                    pseudopotential = label + '.psf'
                else:
                    label = '.'.join([symbol, self.pseudo_qualifier()])
                    pseudopotential = label + '.psf'
            else:
                pseudopotential = spec['pseudopotential']
                label = os.path.basename(pseudopotential)
                label = '.'.join(label.split('.')[:-1])

            if not os.path.isabs(pseudopotential):
                pseudopotential = join(pseudo_path, pseudopotential)

            if not os.path.exists(pseudopotential):
                mess = "Pseudopotential '%s' not found" % pseudopotential
                raise RuntimeError(mess)

            name = os.path.basename(pseudopotential)
            name = name.split('.')
            name.insert(-1, str(species_number))
            if spec['ghost']:
                name.insert(-1, 'ghost')
                atomic_number = -atomic_number
            name = '.'.join(name)

            if join(os.getcwd(), name) != pseudopotential:
                if islink(name) or isfile(name):
                    os.remove(name)
                os.symlink(pseudopotential, name)

            if not spec['excess_charge'] is None:
                atomic_number += 200
                n_atoms = sum(np.array(species_numbers) == species_number)

                paec = float(spec['excess_charge']) / n_atoms
                vc = get_valence_charge(pseudopotential)
                fraction = float(vc + paec) / vc
                pseudo_head = name[:-4]
                fractional_command = os.environ['SIESTA_UTIL_FRACTIONAL']
                cmd = '%s %s %.7f' % (fractional_command,
                                      pseudo_head,
                                      fraction)
                os.system(cmd)

                pseudo_head += '-Fraction-%.5f' % fraction
                synth_pseudo = pseudo_head + '.psf'
                synth_block_filename = pseudo_head + '.synth'
                os.remove(name)
                shutil.copyfile(synth_pseudo, name)
                synth_block = read_vca_synth_block(
                    synth_block_filename,
                    species_number=species_number)
                synth_blocks.append(synth_block)

            if len(synth_blocks) > 0:
                f.write(format_fdf('SyntheticAtoms', list(synth_blocks)))

            label = '.'.join(np.array(name.split('.'))[:-1])
            string = '    %d %d %s' % (species_number, atomic_number, label)
            chemical_labels.append(string)
            if isinstance(spec['basis_set'], PAOBasisBlock):
                pao_basis.append(spec['basis_set'].script(label))
            else:
                basis_sizes.append(("    " + label, spec['basis_set']))
        f.write((format_fdf('ChemicalSpecieslabel', chemical_labels)))
        f.write('\n')
        f.write((format_fdf('PAO.Basis', pao_basis)))
        f.write((format_fdf('PAO.BasisSizes', basis_sizes)))
        f.write('\n')

    def pseudo_qualifier(self):
        """Get the extra string used in the middle of the pseudopotential.
        The retrieved pseudopotential for a specific element will be
        'H.xxx.psf' for the element 'H' with qualifier 'xxx'. If qualifier
        is set to None then the qualifier is set to functional name.
        """
        if self['pseudo_qualifier'] is None:
            return self['xc'][0].lower()
        else:
            return self['pseudo_qualifier']

    def read_results(self):
        """Read the results.
        """
        self.read_number_of_grid_points()
        self.read_energy()
        self.read_forces_stress()
        self.read_eigenvalues()
        self.read_dipole()
        self.read_pseudo_density()
        self.read_hsx()
        self.read_dim()
        if self.results['hsx'] is not None:
            self.read_pld(self.results['hsx'].norbitals,
                          self.atoms.get_number_of_atoms())
            self.atoms.cell = self.results['pld'].cell * Bohr
        else:
            self.results['pld'] = None

        self.read_wfsx()
        self.read_ion(self.atoms)

    def read_ion(self, atoms):
        """Read the ion.xml file of each specie
        """
        from ase.calculators.siesta.import_ion_xml import get_ion

        species, species_numbers = self.species(atoms)

        self.results['ion'] = {}
        for species_number, spec in enumerate(species):
            species_number += 1

            symbol = spec['symbol']
            atomic_number = atomic_numbers[symbol]

            if spec['pseudopotential'] is None:
                if self.pseudo_qualifier() == '':
                    label = symbol
                    pseudopotential = label + '.psf'
                else:
                    label = '.'.join([symbol, self.pseudo_qualifier()])
                    pseudopotential = label + '.psf'
            else:
                pseudopotential = spec['pseudopotential']
                label = os.path.basename(pseudopotential)
                label = '.'.join(label.split('.')[:-1])

            name = os.path.basename(pseudopotential)
            name = name.split('.')
            name.insert(-1, str(species_number))
            if spec['ghost']:
                name.insert(-1, 'ghost')
                atomic_number = -atomic_number
            name = '.'.join(name)

            label = '.'.join(np.array(name.split('.'))[:-1])

            if label not in self.results['ion']:
                fname = label + '.ion.xml'
                self.results['ion'][label] = get_ion(fname)

    def read_hsx(self):
        """
        Read the siesta HSX file.
        return a namedtuple with the following arguments:
        'norbitals', 'norbitals_sc', 'nspin', 'nonzero',
        'is_gamma', 'sc_orb2uc_orb', 'row2nnzero', 'sparse_ind2column',
        'H_sparse', 'S_sparse', 'aB2RaB_sparse', 'total_elec_charge', 'temp'
        """

        import warnings
        from ase.calculators.siesta.import_functions import readHSX

        filename = self.label + '.HSX'
        if isfile(filename):
            self.results['hsx'] = readHSX(filename)
        else:
            warnings.warn(filename + """ does not exist =>
                                     sieta.results["hsx"]=None""",
                                     UserWarning)
            self.results['hsx'] = None

    def read_dim(self):
        """
        Read the siesta DIM file
        Retrun a namedtuple with the following arguments:
        'natoms_sc', 'norbitals_sc', 'norbitals', 'nspin',
        'nnonzero', 'natoms_interacting'
        """

        import warnings
        from ase.calculators.siesta.import_functions import readDIM

        filename = self.label + '.DIM'
        if isfile(filename):
            self.results['dim'] = readDIM(filename)
        else:
            warnings.warn(filename + """does not exist =>
                                     sieta.results["dim"]=None""",
                                     UserWarning)
            self.results['dim'] = None

    def read_pld(self, norb, natms):
        """
        Read the siesta PLD file
        Return a namedtuple with the following arguments:
        'max_rcut', 'orb2ao', 'orb2uorb', 'orb2occ', 'atm2sp',
        'atm2shift', 'coord_sc', 'cell', 'nunit_cells'
        """

        import warnings
        from ase.calculators.siesta.import_functions import readPLD

        filename = self.label + '.PLD'
        if isfile(filename):
            self.results['pld'] = readPLD(filename, norb, natms)
        else:
            warnings.warn(filename + """ does not exist =>
                                     sieta.results["pld"]=None""",
                                     UserWarning)
            self.results['pld'] = None

    def read_wfsx(self):
        """
        Read the siesta WFSX file
        Return a namedtuple with the following arguments:
        """

        import warnings
        from ase.calculators.siesta.import_functions import readWFSX

        if isfile(self.label + '.WFSX'):
            filename = self.label + '.WFSX'
            self.results['wfsx'] = readWFSX(filename)
        elif isfile(self.label + '.fullBZ.WFSX'):
            filename = self.label + '.fullBZ.WFSX'
            readWFSX(filename)
            self.results['wfsx'] = readWFSX(filename)
        else:
            filename = self.label + '.WFSX or ' + self.label + '.fullBZ.WFSX'
            warnings.warn(filename + """ does not exist =>
                                     sieta.results["wfsx"]=None""",
                                     UserWarning)
            self.results['wfsx'] = None

    def read_pseudo_density(self):
        """Read the density if it is there.
        """
        filename = self.label + '.RHO'
        if isfile(filename):
            self.results['density'] = read_rho(filename)

    def read_number_of_grid_points(self):
        """Read number of grid points from SIESTA's text-output file.
        """
        with open(self.label + '.out', 'r') as f:
            for line in f:
                line = line.strip().lower()
                if line.startswith('initmesh: mesh ='):
                    n_points = [int(word) for word in line.split()[3:8:2]]
                    self.results['n_grid_point'] = n_points
                    break
            else:
                raise RuntimeError

    def read_energy(self):
        """Read energy from SIESTA's text-output file.
        """
        with open(self.label + '.out', 'r') as f:
            text = f.read().lower()

        assert 'final energy' in text
        lines = iter(text.split('\n'))

        # Get the energy and free energy the last time it appears
        for line in lines:
            has_energy = line.startswith('siesta: etot    =')
            if has_energy:
                self.results['energy'] = float(line.split()[-1])
                line = next(lines)
                self.results['free_energy'] = float(line.split()[-1])

        if ('energy' not in self.results or
            'free_energy' not in self.results):
            raise RuntimeError

    def read_forces_stress(self):
        """Read the forces and stress from the FORCE_STRESS file.
        """
        with open('FORCE_STRESS', 'r') as f:
            lines = f.readlines()

        stress_lines = lines[1:4]
        stress = np.empty((3, 3))
        for i in range(3):
            line = stress_lines[i].strip().split(' ')
            line = [s for s in line if len(s) > 0]
            stress[i] = [float(s) for s in line]

        self.results['stress'] = np.array(
            [stress[0, 0], stress[1, 1], stress[2, 2],
             stress[1, 2], stress[0, 2], stress[0, 1]])

        self.results['stress'] *= Ry / Bohr**3

        start = 5
        self.results['forces'] = np.zeros((len(lines) - start, 3), float)
        for i in range(start, len(lines)):
            line = [s for s in lines[i].strip().split(' ') if len(s) > 0]
            self.results['forces'][i - start] = [float(s) for s in line[2:5]]

        self.results['forces'] *= Ry / Bohr

    def read_eigenvalues(self):
        """Read eigenvalues from the '.EIG' file.
        This is done pr. kpoint.
        """
        assert os.access(self.label + '.EIG', os.F_OK)
        assert os.access(self.label + '.KP', os.F_OK)

        # Read k point weights
        text = open(self.label + '.KP', 'r').read()
        lines = text.split('\n')
        n_kpts = int(lines[0].strip())
        self.weights = np.zeros((n_kpts,))
        for i in range(n_kpts):
            l = lines[i + 1].split()
            self.weights[i] = float(l[4])

        # Read eigenvalues and fermi-level
        with open(self.label + '.EIG', 'r') as f:
            text = f.read()
        lines = text.split('\n')
        e_fermi = float(lines[0].split()[0])
        tmp = lines[1].split()
        self.n_bands = int(tmp[0])
        n_spin_bands = int(tmp[1])
        self.spin_pol = n_spin_bands == 2
        lines = lines[2:-1]
        lines_per_kpt = (self.n_bands * n_spin_bands / 10 +
                         int((self.n_bands * n_spin_bands) % 10 != 0))
        lines_per_kpt = int(lines_per_kpt)
        eig = dict()
        for i in range(len(self.weights)):
            tmp = lines[i * lines_per_kpt:(i + 1) * lines_per_kpt]
            v = [float(v) for v in tmp[0].split()[1:]]
            for l in tmp[1:]:
                v.extend([float(t) for t in l.split()])
            if self.spin_pol:
                eig[(i, 0)] = np.array(v[0:self.n_bands])
                eig[(i, 1)] = np.array(v[self.n_bands:])
            else:
                eig[(i, 0)] = np.array(v)

        self.results['fermi_energy'] = e_fermi
        self.results['eigenvalues'] = eig

    def read_dipole(self):
        """Read dipole moment.
        """
        dipole = np.zeros([1, 3])
        with open(self.label + '.out', 'r') as f:
            for line in f:
                if line.rfind('Electric dipole (Debye)') > -1:
                    dipole = np.array([float(f) for f in line.split()[5:8]])
        # debye to e*Ang
        self.results['dipole'] = dipole * 0.2081943482534

    def pyscf_tddft(self, Edir=np.array([1.0, 0.0, 0.0]),
                          freq=np.arange(0.0, 10.0, 0.1),
                          units='au',
                          run_tddft=True,
                          save_kernel = True,
                          kernel_name = "tddft_kernel.npy",
                          fname="pol_tensor.npy", 
                          fname_nonin = "noninpol_tensor.npy", **kw):
        """
        Perform TDDFT calculation using the pyscf.nao module for a molecule.

        Parameters
        ----------
        freq: array like
            frequency range for which the polarizability should
            be computed, in eV
        units : str, optional
            unit for the returned polarizability, can be au (atomic units)
            or nm**2
        run_tddft: to run the tddft_calculation or not
        fname: str
            Name of input file name for polariazbility tensor.
            if run_tddft is True: output file
            if run_tddft is False: input file

        kw: keywords for the tddft_iter function from pyscf

        Returns
        -------
            Add to the self.results dict the following items:
        freq range: array like
            array of dimension (nff) containing the frequency range in eV.

        polarizability nonin: array like (complex)
            array of dimension (nff, 3, 3) with nff the frequency number,
            the second and third dimension are the matrix elements of the
            non-interactive polarizability::

                P_xx, P_xy, P_xz, Pyx, .......


        polarizability: array like (complex)
            array of dimension (nff, 3, 3) with nff the frequency number,
            the second and third dimension are the matrix elements of the
            interactive polarizability::

                P_xx, P_xy, P_xz, Pyx, .......

        density change nonin: array like (complex)
            contains the non interacting density change in product basis

        density change inter: array like (complex)
            contains the interacting density change in product basis

        References
        ----------
        https://github.com/cfm-mpc/pyscf/tree/nao

        Example
        -------
        from ase.units import Ry, eV, Ha
        from ase.calculators.siesta import Siesta
        from ase import Atoms
        import numpy as np
        import matplotlib.pyplot as plt

        # Define the systems
        Na8 = Atoms('Na8',
                     positions=[[-1.90503810, 1.56107288, 0.00000000],
                                [1.90503810, 1.56107288, 0.00000000],
                                [1.90503810, -1.56107288, 0.00000000],
                                [-1.90503810, -1.56107288, 0.00000000],
                                [0.00000000, 0.00000000, 2.08495836],
                                [0.00000000, 0.00000000, -2.08495836],
                                [0.00000000, 3.22798122, 2.08495836],
                                [0.00000000, 3.22798122, -2.08495836]],
                     cell=[20, 20, 20])

        # Siesta input
        siesta = Siesta(
                    mesh_cutoff=150 * Ry,
                    basis_set='DZP',
                    pseudo_qualifier='',
                    energy_shift=(10 * 10**-3) * eV,
                    fdf_arguments={
                        'SCFMustConverge': False,
                        'COOP.Write': True,
                        'WriteDenchar': True,
                        'PAO.BasisType': 'split',
                        'DM.Tolerance': 1e-4,
                        'DM.MixingWeight': 0.01,
                        'MaxSCFIterations': 300,
                        'DM.NumberPulay': 4,
                        'XML.Write': True})

        Na8.set_calculator(siesta)
        e = Na8.get_potential_energy()
        freq, pol = siesta.get_polarizability_pyscf_inter(label="siesta",
                                                          jcutoff=7,
                                                          iter_broadening=0.15/Ha,
                                                          xc_code='LDA,PZ',
                                                          tol_loc=1e-6,
                                                          tol_biloc=1e-7,
                                                          freq = np.arange(0.0, 5.0, 0.05))
        # plot polarizability
        plt.plot(freq, pol[:, 0, 0].imag)
        plt.show()
        """

        from ase.calculators.siesta.mbpt_lcao_utils import pol2cross_sec
        assert units in ["nm**2", "au"]

        if run_tddft:
            from pyscf.nao import tddft_iter
            from ase.units import Ha

            tddft = tddft_iter(**kw)
            if save_kernel:
                np.save(kernel_name, tddft.kernel)

            omegas = freq / Ha + 1j * tddft.eps
            tddft.comp_dens_nonin_along_Eext(omegas, Eext=Edir)
            tddft.comp_dens_inter_along_Eext(omegas, Eext=Edir)

            # save polarizability tensor and density change to files
            self.results["freq range"] = freq
            self.results['polarizability nonin'] = np.zeros((freq.size, 3, 3),
                                                dtype=tddft.p0_mat.dtype)
            self.results['polarizability inter'] = np.zeros((freq.size, 3, 3),
                                                dtype=tddft.p_mat.dtype)
            self.results["density change nonin"] = tddft.dn0
            self.results["density change inter"] = tddft.dn
            for xyz1 in range(3):
                for xyz2 in range(3):
                    if units == 'nm**2':
                        p0 = pol2cross_sec(-tddft.p0_mat[xyz1, xyz2, :],
                                          freq)
                        p = pol2cross_sec(-tddft.p_mat[xyz1, xyz2, :],
                                          freq)
                        self.results['polarizability nonin'][:, xyz1, xyz2] = p0
                        self.results['polarizability inter'][:, xyz1, xyz2] = p
                    else:
                        self.results['polarizability nonin'][:, xyz1, xyz2] = \
                                                -tddft.p0_mat[xyz1, xyz2, :]
                        self.results['polarizability inter'][:, xyz1, xyz2] = \
                                                -tddft.p_mat[xyz1, xyz2, :]

        else:
            # load polarizability tensor from previous calculations
            p0_mat = np.load(fname_nonin)
            p_mat = np.load(fname)

            self.results['polarizability nonin'] = np.zeros((freq.size, 3, 3),
                                                        dtype=p0_mat.dtype)
            self.results['polarizability inter'] = np.zeros((freq.size, 3, 3),
                                                        dtype=p_mat.dtype)

            for xyz1 in range(3):
                for xyz2 in range(3):
                    if units == 'nm**2':
                        p0 = pol2cross_sec(-p0_mat[xyz1, xyz2, :], freq)
                        p = pol2cross_sec(-p_mat[xyz1, xyz2, :], freq)

                        self.results['polarizability nonin'][:, xyz1, xyz2] = p0
                        self.results['polarizability inter'][:, xyz1, xyz2] = p
                    else:
                        self.results['polarizability nonin'][:, xyz1, xyz2] = \
                                                        -p0_mat[xyz1, xyz2, :]
                        self.results['polarizability inter'][:, xyz1, xyz2] = \
                                                        -p_mat[xyz1, xyz2, :]

    def pyscf_tddft_eels(self, velec = np.array([20.0, 0.0, 0.0]),
                               b = np.array([0.0, 0.0, 0.0]),
                               freq=np.arange(0.0, 10.0, 0.1),
                               tddft = None,
                               save_kernel = True,
                               kernel_name = "tddft_kernel.npy",
                               tmp_fname = None,
                               **kw):
        """
        Perform TDDFT calculation using the pyscf.nao module for a molecule.
        The external pertubation is created by a electron moving at the velocity velec
        and with an impact parameter b

        Parameters
        ----------
        freq: array like
            frequency range for which the polarizability should
            be computed, in eV
        velec: array like
            velocity vector of the projectile
        b: array like
            offset vector of the projectile
        tddft: tddft_tem class from a previous calculation
        save_kernel: save the kernel for future use
        kernel_name: name of the file for the kernel
        tmp_fname: temporary name to save the eels spectra while running the calculations
        kw: keywords for the tddft_tem function from pyscf

        Returns
        -------
        tddft:
            if running pyscf_tddft_eels in a loop over the velocity or the 
            impact parameter, there is no point to initialize again the tddft
            calculation (vertex and kernel will be the same)

            Add to the self.results dict the following items:
        freq range: array like
            array of dimension (nff) containing the frequency range in eV.

        eel spectra nonin: array like (complex)
            array of dimension (nff) with nff the frequency number,


        eel spectra inter: array like (complex)
            array of dimension (nff) with nff the frequency number,

        density change eels nonin: array like (complex)
            contains the non interacting density change in product basis

        density change eels inter: array like (complex)
            contains the interacting density change in product basis

        References
        ----------
        https://github.com/cfm-mpc/pyscf/tree/nao

        Example
        -------
        from ase.units import Ry, eV, Ha
        from ase.calculators.siesta import Siesta
        from ase import Atoms
        import numpy as np
        import matplotlib.pyplot as plt

        # Define the systems
        Na8 = Atoms('Na8',
                    positions=[[-1.90503810, 1.56107288, 0.00000000],
                               [1.90503810, 1.56107288, 0.00000000],
                               [1.90503810, -1.56107288, 0.00000000],
                               [-1.90503810, -1.56107288, 0.00000000],
                               [0.00000000, 0.00000000, 2.08495836],
                               [0.00000000, 0.00000000, -2.08495836],
                               [0.00000000, 3.22798122, 2.08495836],
                               [0.00000000, 3.22798122, -2.08495836]],
                    cell=[20, 20, 20])

        # enter siesta input
        siesta = Siesta(
            mesh_cutoff=150 * Ry,
            basis_set='DZP',
            pseudo_qualifier='',
            energy_shift=(10 * 10**-3) * eV,
            fdf_arguments={
                'SCFMustConverge': False,
                'COOP.Write': True,
                'WriteDenchar': True,
                'PAO.BasisType': 'split',
                'DM.Tolerance': 1e-4,
                'DM.MixingWeight': 0.01,
                'MaxSCFIterations': 300,
                'DM.NumberPulay': 4,
                'XML.Write': True})


        Na8.set_calculator(siesta)
        e = Na8.get_potential_energy()
        tddft = siesta.pyscf_tddft_eels(label="siesta", jcutoff=7, iter_broadening=0.15/Ha,
                    xc_code='LDA,PZ', tol_loc=1e-6, tol_biloc=1e-7, freq = np.arange(0.0, 5.0, 0.05))

        # plot eel spectra
        fig = plt.figure(1)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.plot(siesta.results["freq range"], siesta.results["eel spectra nonin"].imag)
        ax2.plot(siesta.results["freq range"], siesta.results["eel spectra inter"].imag)

        ax1.set_xlabel(r"$\omega$ (eV)")
        ax2.set_xlabel(r"$\omega$ (eV)")

        ax1.set_ylabel(r"Im($P_{xx}$) (au)")
        ax2.set_ylabel(r"Im($P_{xx}$) (au)")

        ax1.set_title(r"Non interacting")
        ax2.set_title(r"Interacting")

        fig.tight_layout()

        plt.show()

        """


        from pyscf.nao import tddft_tem
        from ase.units import Ha

        assert velec.size == 3
        assert b.size == 3

        if tddft is None:
            self.results["freq range"] = freq
            omegas = freq / Ha

            # for eels, omega is real array
            tddft = tddft_tem(freq = omegas, **kw)
            if save_kernel:
                np.save(kernel_name, tddft.kernel)


        self.results['eel spectra nonin'] = tddft.get_spectrum_nonin(velec=velec,
                                                                  beam_offset = b, 
                                                                  tmp_fname=tmp_fname)

        self.results['eel spectra inter'] = tddft.get_spectrum_inter(velec=velec,
                                                                  beam_offset = b,
                                                                  tmp_fname=tmp_fname)

        self.results["density change eels nonin"] = tddft.dn0
        self.results["density change eels inter"] = tddft.dn

        return tddft


    def get_polarizability_mbpt(self, mbpt_inp=None,
                                output_name='mbpt_lcao.out',
                                format_output='hdf5', units='au'):
        """
        Warning!!
            Out dated version, try get_polarizability_pyscf


        Calculate the polarizability by running the mbpt_lcao program.
        The mbpt_lcao program need the siesta output, therefore siesta need
        to be run first.

        Parameters
        ----------
        mbpt_inp : dict, optional
            dictionnary of the input for the mbpt_lcao program
            (http://mbpt-domiprod.wikidot.com/list-of-parameters)
            if mbpt_inp is None, the function read the output file
            from a previous mbpt_lcao run.
        output_name : str, optional
            Name of the mbpt_lcao output
        format_output : str, optional
            Format of the mbpt_lcao output data,
            if hdf5, the output name is tddft_iter_output.hdf5 if
            do_tddft_iter is set to 1 the output name is
            tddft_tem_output.hdf5 if do_tddft_tem is set to 1
            if txt, a lot of output data files are produced depending on
            the input, in the text and fortran binaries format
        units : str, optional
            unit for the returned polarizability, can be au (atomic units)
            or nm**2

        Returns
        -------
        freq : array like
            array of dimension (nff) containing the frequency range in eV.

        self.results['polarizability'], array like
            array of dimension (nff, 3, 3) with nff the frequency number,
            the second and third dimension are the matrix elements of the
            polarizability::

                P_xx, P_xy, P_xz, Pyx, .......

        References
        ----------
        http://mbpt-domiprod.wikidot.com

        Example
        -------
        import os
        from ase.units import Ry, eV
        from ase.calculators.siesta import Siesta
        from ase import Atoms
        import numpy as np
        import matplotlib.pyplot as plt

        #Define the systems
        Na8 = Atoms('Na8',
        positions=[[-1.90503810, 1.56107288, 0.00000000],
                    [1.90503810, 1.56107288, 0.00000000],
                    [1.90503810, -1.56107288, 0.00000000],
                    [-1.90503810, -1.56107288, 0.00000000],
                    [0.00000000, 0.00000000, 2.08495836],
                    [0.00000000, 0.00000000, -2.08495836],
                    [0.00000000, 3.22798122, 2.08495836],
                    [0.00000000, 3.22798122, -2.08495836]],
                    cell=[20, 20, 20])

        #enter siesta input
        siesta = Siesta(
            mesh_cutoff=150 * Ry,
            basis_set='DZP',
            pseudo_qualifier='',
            energy_shift=(10 * 10**-3) * eV,
            fdf_arguments={
            'SCFMustConverge': False,
            'COOP.Write': True,
            'WriteDenchar': True,
            'PAO.BasisType': 'split',
            'DM.Tolerance': 1e-4,
            'DM.MixingWeight': 0.01,
            'MaxSCFIterations': 300,
            'DM.NumberPulay': 4})


        #mbpt_lcao input
        mbpt_inp = {'prod_basis_type' : 'MIXED',
                    'solver_type' : 1,
                    'gmres_eps' : 0.001,
                    'gmres_itermax':256,
                    'gmres_restart':250,
                    'gmres_verbose':20,
                    'xc_ord_lebedev':14,
                    'xc_ord_gl':48,
                    'nr':512,
                    'akmx':100,
                    'eigmin_local':1e-06,
                    'eigmin_bilocal':1e-08,
                    'freq_eps_win1':0.15,
                    'd_omega_win1':0.05,
                    'dt':0.1,
                    'omega_max_win1':5.0,
                    'ext_field_direction':2,
                    'dr':np.array([0.3, 0.3, 0.3]),
                    'para_type':'MATRIX',
                    'chi0_v_algorithm':14,
                    'format_output':'text',
                    'comp_dens_chng_and_polarizability':1,
                    'store_dens_chng':1,
                    'enh_given_volume_and_freq':0,
                    'diag_hs':0,
                    'do_tddft_tem':0,
                    'do_tddft_iter':1,
                    'plot_freq':3.02,
                    'gwa_initialization':'SIESTA_PB'}


        Na8.set_calculator(siesta)
        e = Na8.get_potential_energy() #run siesta
        freq, pol = siesta.get_polarizability_siesta(mbpt_inp,
                                                     format_output='txt',
                                                     units='nm**2')

        #plot polarizability
        plt.plot(freq, pol[:, 0, 0])

        plt.show()
        """
        from ase.calculators.siesta.mbpt_lcao import MBPT_LCAO
        from ase.calculators.siesta.mbpt_lcao_io import read_mbpt_lcao_output
        import warnings

        warnings.warn("Out dated version, try get_polarizability_pyscf")

        if mbpt_inp is not None:
            tddft = MBPT_LCAO(mbpt_inp)
            tddft.run_mbpt_lcao(output_name, True)

        r = read_mbpt_lcao_output()

        r.args.format_input = format_output

        # read real part
        r.args.ReIm = 're'
        data = r.Read()
        self.results['polarizability'] = data.Array

        # read imaginary part
        r.args.ReIm = 'im'
        data = r.Read()
        self.results['polarizability'] = (self.results['polarizability'] +
                                          complex(0.0, 1.0) * data.Array)

        if units == 'nm**2':
            from ase.calculators.siesta.mbpt_lcao_utils import pol2cross_sec
            for i in range(2):
                for j in range(2):
                    p = pol2cross_sec(self.results['polarizability'][:, i, j],
                                      data.freq)
                    self.results['polarizability'][:, i, j] = p

            print('unit nm**2')
            # self.results['polarizability'] = data.Array
        elif units == 'au':
            print('unit au')
            # self.results['polarizability'] = data.Array
        else:
            raise ValueError('units can be only au or nm**2')

        return data.freq, self.results['polarizability']
