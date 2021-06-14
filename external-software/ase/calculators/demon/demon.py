from __future__ import print_function
"""This module defines an ASE interface to deMon.

http://www.demon-software.com

"""
import os
import os.path as op
import subprocess
import pickle
import shutil

import numpy as np

from ase.units import Bohr, Hartree
import ase.data
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.calculator import equal
import ase.io
from .demon_io import parse_xray

m_e_to_amu = 1822.88839


class Parameters_deMon(Parameters):
    """Parameters class for the calculator.
    Documented in Base_deMon.__init__

    The options here are the most important ones that the user needs to be
    aware of. Further options accepted by deMon can be set in the dictionary
    input_arguments.

    """
    def __init__(
            self,
            label='rundir',
            atoms=None,
            command=None,
            restart=None,
            basis_path=None,
            ignore_bad_restart_file=False,
            deMon_restart_path='.',
            title='deMon input file',
            scftype='RKS',
            forces=False,
            dipole=False,
            xc='VWN',
            guess='TB',
            print_out='MOE',
            basis={},
            ecps={},
            mcps={},
            auxis={},
            augment={},
            input_arguments=None):
        kwargs = locals()
        kwargs.pop('self')
        Parameters.__init__(self, **kwargs)


class Demon(FileIOCalculator):
    """Calculator interface to the deMon code. """

    implemented_properties = (
        'energy',
        'forces',
        'dipole',
        'eigenvalues')

    def __init__(self, **kwargs):
        """ASE interface to the deMon code.
        
        The deMon2k code can be obtained from http://www.demon-software.com

        The DEMON_COMMAND environment variable must be set to run the executable, in bash it would be set along the lines of
        export DEMON_COMMAND="deMon.4.3.6.std > deMon_ase.out 2>&1"

        Parameters:

        label : str 
            relative path to the run directory
        atoms : Atoms object
            the atoms object
        command  : str
            Command to run deMon. If not present the environment varable DEMON_COMMAND will be used
        restart  : str
            Relative path to ASE restart directory for parameters and atoms object and results
        basis_path  : str 
            Relative path to the directory containing BASIS, AUXIS, ECPS, MCPS and AUGMENT
        ignore_bad_restart_file : bool 
            Ignore broken or missing ASE restart files
            By default, it is an error if the restart
            file is missing or broken.
        deMon_restart_path  : str 
            Relative path to the deMon restart dir
        title : str 
            Title in the deMon input file.
        scftype : str 
            Type of scf
        forces  : bool 
            If True a force calculation will be enforced.
        dipole  : bool
            If True a dipole calculation will be enforced
        xc : str 
            xc-functional
        guess : str 
            guess for initial density and wave functions
        print_out : str | list 
            Options for the printing in deMon
        basis : dict 
            Definition of basis sets.
        ecps  : dict 
            Definition of ECPs
        mcps  : dict
            Definition of MCPs
        auxis  : dict 
            Definition of AUXIS
        augment : dict
            Definition of AUGMENT
        input_arguments : dict 
            Explicitly given input arguments. The key is the input keyword
            and the value is either a str, a list of str (will be written on the same line as the keyword),
            or a list of lists of str (first list is written on the first line, the others on following lines.)
        
        For example usage, see the tests h2o.py and h2o_xas_xes.py in the directory ase/test/demon
        
        """
        
        parameters = Parameters_deMon(**kwargs)
        
        # Setup the run command
        command = parameters['command']
        if command is None:
            command = os.environ.get('DEMON_COMMAND')

        if command is None:
            mess = 'The "DEMON_COMMAND" environment is not defined.'
            raise ValueError(mess)
        else:
            parameters['command'] = command
            
        # Call the base class.
        FileIOCalculator.__init__(
            self,
            **parameters)

    def __getitem__(self, key):
        """Convenience method to retrieve a parameter as
        calculator[key] rather than calculator.parameters[key]

            Parameters:
                key       : str, the name of the parameters to get.
        """
        return self.parameters[key]

    def set(self, **kwargs):
        """Set all parameters.

        Parameters:
            kwargs  : Dictionary containing the keywords for deMon
        """
        # Put in the default arguments.
        kwargs = self.default_parameters.__class__(**kwargs)

        if 'parameters' in kwargs:
            filename = kwargs.pop('parameters')
            parameters = Parameters.read(filename)
            parameters.update(kwargs)
            kwargs = parameters

        changed_parameters = {}

        for key, value in kwargs.items():
            oldvalue = self.parameters.get(key)
            if key not in self.parameters or not equal(value, oldvalue):
                changed_parameters[key] = value
                self.parameters[key] = value

        return changed_parameters

    def link_file(self, fromdir, todir, filename):

        if op.exists(todir + '/' + filename):
            os.remove(todir + '/' + filename)
                
        if op.exists(fromdir + '/' + filename):
            os.symlink(fromdir + '/' + filename, 
                       todir + '/' + filename)
        else:
            raise RuntimeError(
                "{0} doesn't exist".format(fromdir + '/' + filename))




    def calculate(self,
                  atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        """Capture the RuntimeError from FileIOCalculator.calculate
        and add a little debug information from the deMon output.

        See base FileIocalculator for documentation.
        """

        if atoms is not None:
            self.atoms = atoms.copy()

        self.write_input(self.atoms, properties, system_changes)
        if self.command is None:
            raise RuntimeError('Please set $%s environment variable ' %
                               ('DEMON_COMMAND') +
                               'or supply the command keyword')
        command = self.command  # .replace('PREFIX', self.prefix)
        olddir = os.getcwd()

        # basis path
        basis_path = self.parameters['basis_path']
        if basis_path is None:
            basis_path = os.environ.get('DEMON_BASIS_PATH')

        if basis_path is None:
            raise RuntimeError('Please set basis_path keyword,' +
                               ' or the DEMON_BASIS_PATH' +
                               ' environment variable')

        try:
            # link restart file
            value = self.parameters['guess']
            if value.upper() == 'RESTART':
                value2 = self.parameters['deMon_restart_path']

                if op.exists(self.directory + '/deMon.rst')\
                        or op.islink(self.directory + '/deMon.rst'):
                    os.remove(self.directory + '/deMon.rst')
                abspath = op.abspath(value2)
                
                if op.exists(abspath + '/deMon.mem') \
                        or op.islink(abspath + '/deMon.mem'):

                    shutil.copy(abspath + '/deMon.mem',
                                self.directory + '/deMon.rst')
                else:
                    raise RuntimeError(
                        "{0} doesn't exist".format(abspath + '/deMon.rst'))


            abspath = op.abspath(basis_path)

            # link basis
            self.link_file(abspath, self.directory, 'BASIS')

            # link auxis
            self.link_file(abspath, self.directory, 'AUXIS')

            # link ecps
            self.link_file(abspath, self.directory, 'ECPS')            

            # link mcps
            self.link_file(abspath, self.directory, 'MCPS')            

            # link ffds
            self.link_file(abspath, self.directory, 'FFDS')            

            # go to directory and run calculation
            os.chdir(self.directory)
            errorcode = subprocess.call(command, shell=True)
        finally:
            os.chdir(olddir)

        if errorcode:
            raise RuntimeError('%s returned an error: %d' %
                               (self.name, errorcode))

        try:
            self.read_results()
        except:
            with open(self.directory + '/deMon.out', 'r') as f:
                lines = f.readlines()
            debug_lines = 10
            print('##### %d last lines of the deMon.out' % debug_lines)
            for line in lines[-20:]:
                print(line.strip())
            print('##### end of deMon.out')
            raise RuntimeError
        



    def set_label(self, label):
        """Set label directory """

        self.label = label

        # in our case self.directory = self.label
        self.directory = self.label
        if self.directory == '':
            self.directory = os.curdir

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write input (in)-file.
        See calculator.py for further details.

        Parameters:
             atoms        : The Atoms object to write.
             properties   : The properties which should be calculated.
             system_changes : List of properties changed since last run.
        
        """
        # Call base calculator.
        FileIOCalculator.write_input(
            self,
            atoms=atoms,
            properties=properties,
            system_changes=system_changes)

        if system_changes is None and properties is None:
            return

        filename = self.label + '/deMon.inp'

        add_print = ''

        # Start writing the file.
        with open(filename, 'w') as f:

            # write keyword argument keywords
            value = self.parameters['title']
            self._write_argument('TITLE', value, f)

            f.write('#\n')
            
            value = self.parameters['scftype']
            self._write_argument('SCFTYPE', value, f)

            value = self.parameters['xc']
            self._write_argument('VXCTYPE', value, f)

            value = self.parameters['guess']
            self._write_argument('GUESS', value, f)

            # obtain forces through a single BOMD step
            # only if forces is in properties, or if keyword forces is True
            value = self.parameters['forces']
            if 'forces' in properties or value:

                self._write_argument('DYNAMICS',
                                     ['INT=1', 'MAX=0', 'STEP=0'], f)
                self._write_argument('TRAJECTORY', 'FORCES', f)
                self._write_argument('VELOCITIES', 'ZERO', f)
                add_print = add_print + ' ' + 'MD OPT'

            # if dipole is True, enforce dipole calculation.
            # Otherwise only if asked for
            value = self.parameters['dipole']
            if 'dipole' in properties or value:
                self._write_argument('DIPOLE', '', f)

            # print argument, here other options could change this
            value = self.parameters['print_out']
            assert(type(value) is str)
            value = value + add_print

            if not len(value) == 0:
                self._write_argument('PRINT', value, f)
                f.write('#\n')

            # write general input arguments
            self._write_input_arguments(f)
            
            f.write('#\n')

            # write basis set, ecps, mcps, auxis, augment
            basis = self.parameters['basis']
            if 'all' not in basis:
                basis['all'] = 'DZVP'
            self._write_basis(f, atoms, basis, string='BASIS')

            ecps = self.parameters['ecps']
            if not len(ecps) == 0:
                self._write_basis(f, atoms, ecps, string='ECPS')

            mcps = self.parameters['mcps']
            if not len(mcps) == 0:
                self._write_basis(f, atoms, mcps, string='MCPS')

            auxis = self.parameters['auxis']
            if not len(auxis) == 0:
                self._write_basis(f, atoms, auxis, string='AUXIS')

            augment = self.parameters['augment']
            if not len(augment) == 0:
                self._write_basis(f, atoms, augment, string='AUGMENT')

            # write geometry
            self._write_atomic_coordinates(f, atoms)

            # write pickle of Parameters
            pickle.dump(self.parameters,
                        open(self.label + '/deMon_parameters.pckl', 'wb'))

            # write xyz file for good measure.
            ase.io.write(self.label + '/deMon_atoms.xyz', self.atoms)
                
    def read(self, restart_path):
        """Read parameters from directory restart_path."""

        self.set_label(restart_path)

        if not op.exists(restart_path + '/deMon.inp'):
            raise ReadError('The restart_path file {0} does not exist'
                            .format(restart_path))

        if op.exists(restart_path + '/deMon_parameters.pckl'):
            parameters = pickle.load(open(restart_path +
                                          '/deMon_parameters.pckl', 'r'))
            self.parameters = parameters

        self.atoms = self.deMon_inp_to_atoms(restart_path + '/deMon.inp')
        
        self.read_results()

    def _write_input_arguments(self, f):
        """Write directly given input-arguments."""
        input_arguments = self.parameters['input_arguments']

        # Early return
        if input_arguments is None:
            return

        for key, value in input_arguments.items():
            self._write_argument(key, value, f)

    def _write_argument(self, key, value, f):
        """Write an argument to file.
         key :  a string coresponding to the input keyword
         value : the arguemnts, can be a string, a number or a list
         f :  and open file
        """
        
        # for only one argument, write on same line
        if not isinstance(value, (tuple, list)):
            line = key.upper()
            line += '    ' + str(value).upper()
            f.write(line)
            f.write('\n')

        # for a list, write first argument on the first line,
        # then the rest on new lines
        else:
            line = key
            if not isinstance(value[0], (tuple, list)):
                for i in range(len(value)):
                    line += '  ' + str(value[i].upper())
                f.write(line)
                f.write('\n')
            else:
                for i in range(len(value)):
                    for j in range(len(value[i])):
                        line += '  ' + str(value[i][j]).upper()
                    f.write(line)
                    f.write('\n')
                    line = ''
                        
    def _write_atomic_coordinates(self, f, atoms):
        """Write atomic coordinates.
        
        Parameters:
        - f:     An open file object.
        - atoms: An atoms object.
        """

        f.write('#\n')
        f.write('# Atomic coordinates\n')
        f.write('#\n')
        f.write('GEOMETRY CARTESIAN ANGSTROM\n')

        for i in range(len(atoms)):
            xyz = atoms.get_positions()[i]
            chem_symbol = atoms.get_chemical_symbols()[i]
            chem_symbol += str(i + 1)
            
            # if tag is set to 1 then we have a ghost atom,
            # set nuclear charge to 0
            if(atoms.get_tags()[i] == 1):
                nuc_charge = str(0)
            else:
                nuc_charge = str(atoms.get_atomic_numbers()[i])
            
            mass = atoms.get_masses()[i]
                
            line = '{0:6s}'.format(chem_symbol).rjust(10) + ' '
            line += '{0:.5f}'.format(xyz[0]).rjust(10) + ' '
            line += '{0:.5f}'.format(xyz[1]).rjust(10) + ' '
            line += '{0:.5f}'.format(xyz[2]).rjust(10) + ' '
            line += '{0:5s}'.format(nuc_charge).rjust(10) + ' '
            line += '{0:.5f}'.format(mass).rjust(10) + ' '
            
            f.write(line)
            f.write('\n')

    # routine to write basis set inormation, including ecps and auxis
    def _write_basis(self, f, atoms, basis={}, string='BASIS'):
        """Write basis set, ECPs, AUXIS, or AUGMENT basis
        
        Parameters:
        - f:     An open file object.
        - atoms: An atoms object.
        - basis: A dictionary specifying the basis set
        - string: 'BASIS', 'ECP','AUXIS' or 'AUGMENT'
        """

        # basis for all atoms
        line = '{0}'.format(string).ljust(10)

        if 'all' in basis:
            default_basis = basis['all']
            line += '({0})'.format(default_basis).rjust(16)
        
        f.write(line)
        f.write('\n')

        # basis for all atomic species
        chemical_symbols = atoms.get_chemical_symbols()
        chemical_symbols_set = set(chemical_symbols)

        for i in range(chemical_symbols_set.__len__()):
            symbol = chemical_symbols_set.pop()

            if symbol in basis:
                line = '{0}'.format(symbol).ljust(10)
                line += '({0})'.format(basis[symbol]).rjust(16)
                f.write(line)
                f.write('\n')

        # basis for individual atoms
        for i in range(len(atoms)):
            
            if i in basis:
                symbol = str(chemical_symbols[i])
                symbol += str(i + 1)

                line = '{0}'.format(symbol).ljust(10)
                line += '({0})'.format(basis[i]).rjust(16)
                f.write(line)
                f.write('\n')

    # Analysis routines
    def read_results(self):
        """Read the results from output files."""
        self.read_energy()
        self.read_forces(self.atoms)
        self.read_eigenvalues()
        self.read_dipole()
        self.read_xray()
        
    def read_energy(self):
        """Read energy from deMon's text-output file."""
        with open(self.label + '/deMon.out', 'r') as f:
            text = f.read().upper()

        lines = iter(text.split('\n'))

        for line in lines:
            if line.startswith(' TOTAL ENERGY                ='):
                self.results['energy'] = float(line.split()[-1]) * Hartree
                break
        else:
            raise RuntimeError

    def read_forces(self, atoms):
        """Read the forces from the deMon.out file."""
    
        natoms = len(atoms)
        filename = self.label + '/deMon.out'

        if op.isfile(filename):
            with open(filename, 'r') as f:
                lines = f.readlines()

                # find line where the orbitals start
                flag_found = False
                for i in range(len(lines)):
                    if lines[i].rfind('GRADIENTS OF TIME STEP 0 IN A.U.') > -1:
                        start = i + 4
                        flag_found = True
                        break
            
                if flag_found:
                    self.results['forces'] = np.zeros((natoms, 3), float)
                    for i in range(natoms):
                        line = [s for s in lines[i + start].strip().split(' ')
                                if len(s) > 0]
                        f = -np.array([float(x) for x in line[2:5]])
                        self.results['forces'][i, :] = f * (Hartree / Bohr)

    def read_eigenvalues(self):
        """Read eigenvalues from the 'deMon.out' file."""
        assert os.access(self.label + '/deMon.out', os.F_OK)

        # Read eigenvalues
        with open(self.label + '/deMon.out', 'r') as f:
            lines = f.readlines()

        # try  PRINT MOE
        eig_alpha, occ_alpha = self.read_eigenvalues_one_spin(
            lines, 'ALPHA MO ENERGIES', 6)
        eig_beta, occ_beta = self.read_eigenvalues_one_spin(
            lines, 'BETA MO ENERGIES', 6)

        # otherwise try PRINT MOS
        if len(eig_alpha) == 0 and len(eig_beta) == 0:
            eig_alpha, occ_alpha = self.read_eigenvalues_one_spin(
                lines, 'ALPHA MO COEFFICIENTS', 5)
            eig_beta, occ_beta = self.read_eigenvalues_one_spin(
                lines, 'BETA MO COEFFICIENTS', 5)

        self.results['eigenvalues'] = np.array([eig_alpha, eig_beta]) * Hartree
        self.results['occupations'] = np.array([occ_alpha, occ_beta])
 
    def read_eigenvalues_one_spin(self, lines, string, neigs_per_line):
        """Utility method for retreiving eigenvalues after the string "string"
        with neigs_per_line eigenvlaues written per line
        """
        eig = []
        occ = []

        skip_line = False
        more_eigs = False

        # find line where the orbitals start
        for i in range(len(lines)):
            if lines[i].rfind(string) > -1:
                ii = i
                more_eigs = True
                break

        while more_eigs:
            # search for two empty lines in a row preceeding a line with
            # numbers
            for i in range(ii + 1, len(lines)):
                if len(lines[i].split()) == 0 and \
                        len(lines[i + 1].split()) == 0 and \
                        len(lines[i + 2].split()) > 0:
                    ii = i + 2
                    break

            # read eigenvalues, occupations
            line = lines[ii].split()
            if len(line) < neigs_per_line:
                # last row
                more_eigs = False
            if line[0] != str(len(eig) + 1):
                more_eigs = False
                skip_line = True

            if not skip_line:
                line = lines[ii + 1].split()
                for l in line:
                    eig.append(float(l))
                line = lines[ii + 3].split()
                for l in line:
                    occ.append(float(l))
                ii = ii + 3

        return eig, occ

    def read_dipole(self):
        """Read dipole moment."""
        dipole = np.zeros(3)
        with open(self.label + '/deMon.out', 'r') as f:
            lines = f.readlines()

            for i in range(len(lines)):
                if lines[i].rfind('DIPOLE') > -1 and lines[i].rfind('XAS') == -1:
                    dipole[0] = float(lines[i + 1].split()[3])
                    dipole[1] = float(lines[i + 2].split()[3])
                    dipole[2] = float(lines[i + 3].split()[3])

                    # debye to e*Ang
                    self.results['dipole'] = dipole * 0.2081943482534

                    break
 
    def read_xray(self):
        """Read deMon.xry if present."""


        # try to read core IP from, .out file
        filename = self.label + '/deMon.out'
        core_IP = None
        if op.isfile(filename):
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            for i in range(len(lines)):
                if lines[i].rfind('IONIZATION POTENTIAL') > -1:
                    core_IP = float(lines[i].split()[3])
                    
        try:
            mode, ntrans, E_trans, osc_strength, trans_dip = parse_xray(self.label + '/deMon.xry')
        except ReadError:
            pass 
        else:
            xray_results = {'xray_mode': mode,
                            'ntrans': ntrans,
                            'E_trans': E_trans,
                            'osc_strength': osc_strength,  # units?
                            'trans_dip': trans_dip, # units?
                            'core_IP':core_IP}  
            
            self.results['xray'] = xray_results
 

            
    def deMon_inp_to_atoms(self, filename):
        """Routine to read deMon.inp and convert it to an atoms object."""

        with open(filename, 'r') as f:
            lines = f.readlines()

        # find line where geometry starts
        for i in range(len(lines)):
            if lines[i].rfind('GEOMETRY') > -1:
                if lines[i].rfind('ANGSTROM'):
                    coord_units = 'Ang'
                elif lines.rfind('Bohr'):
                    coord_units = 'Bohr'
                ii = i
                break

        chemical_symbols = []
        xyz = []
        atomic_numbers = []
        masses = []

        for i in range(ii + 1, len(lines)):
            try:
                line = lines[i].split()

                if(len(line) > 0):
                    for symbol in ase.data.chemical_symbols:
                        found = None
                        if line[0].upper().rfind(symbol.upper()) > -1:
                            found = symbol
                            break
                        
                        if found is not None:
                            chemical_symbols.append(found)
                        else:
                            break

                        xyz.append([float(line[1]), float(line[2]), float(line[3])])
                
                if len(line) > 4:
                    atomic_numbers.append(int(line[4]))
                
                if len(line) > 5:
                    masses.append(float(line[5]))

            except:
                raise RuntimeError

        if coord_units == 'Bohr':
            xyz = xyz * Bohr

        natoms = len(chemical_symbols)

        # set atoms object
        atoms = ase.Atoms(symbols=chemical_symbols, positions=xyz)

        # if atomic numbers were read in, set them
        if(len(atomic_numbers) == natoms):
            atoms.set_atomic_numbers(atomic_numbers)
                
        # if masses were read in, set them
        if(len(masses) == natoms):
            atoms.set_masses(masses)
            
        return atoms
