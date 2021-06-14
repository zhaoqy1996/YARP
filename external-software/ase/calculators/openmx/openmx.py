"""
    The ASE Calculator for OpenMX <http://www.openmx-square.org>
    A Python interface to the software package for nano-scale
    material simulations based on density functional theories.
    Copyright (C) 2017 Charles Thomas Johnson, Jae Hwan Shim and JaeJun Yu

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

from __future__ import print_function
import os
import time
import subprocess
import re
import warnings
from distutils.version import LooseVersion
import numpy as np
from ase.geometry import cell_to_cellpar
from ase.calculators.calculator import (FileIOCalculator, Calculator, equal,
                                        all_changes, kptdensity2monkhorstpack,
                                        PropertyNotImplementedError)
from ase.calculators.openmx.parameters import OpenMXParameters
from ase.calculators.openmx.default_settings import default_dictionary
from ase.calculators.openmx.reader import read_openmx, get_file_name
from ase.calculators.openmx.writer import write_openmx
#from ase.calculators.openmx.dos import DOS


class OpenMX(FileIOCalculator):
    """
    Calculator interface to the OpenMX code.
    """

    implemented_properties = (
        'free_energy',       # Same value with energy
        'energy',
        'forces',
        'stress',
        'dipole',
        'chemical_potential',
        'magmom',
        'magmoms',
        'eigenvalues',
    )

    default_parameters = OpenMXParameters()

    default_pbs = {
        'processes': 1,
        'walltime': "10:00:00",
        'threads': 1,
        'nodes': 1
    }

    default_mpi = {
        'processes': 1,
        'threads': 1
    }

    default_output_setting = {
        'nohup': True,
        'debug': False
    }

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='./openmx', atoms=None, command=None, mpi=None,
                 pbs=None, **kwargs):

        # Initialize and put the default parameters.
        self.initialize_pbs(pbs)
        self.initialize_mpi(mpi)
        self.initialize_output_setting(**kwargs)

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, command, **kwargs)

    def __getitem__(self, key):
        """Convenience method to retrieve a parameter as
        calculator[key] rather than calculator.parameters[key]

            Parameters:
                -key       : str, the name of the parameters to get.
        """
        return self.parameters[key]

    def __setitem__(self, key, value):
        self.parameters[key] = value

    def initialize_output_setting(self, **kwargs):
        output_setting = {}
        self.output_setting = dict(self.default_output_setting)
        for key, value in kwargs.items():
            if key in self.default_output_setting:
                output_setting[key] = value
        self.output_setting.update(output_setting)
        self.__dict__.update(self.output_setting)

    def initialize_pbs(self, pbs):
        if pbs:
            self.pbs = dict(self.default_pbs)
            for key in pbs:
                if key not in self.default_pbs:
                    allowed = ', '.join(list(self.default_pbs.keys()))
                    raise TypeError('Unexpected keyword "{0}" in "pbs" '
                                    'dictionary.  Must be one of: {1}'
                                    .format(key, allowed))
            # Put dictionary into python variable
            self.pbs.update(pbs)
            self.__dict__.update(self.pbs)
        else:
            self.pbs = None

    def initialize_mpi(self, mpi):
        if mpi:
            self.mpi = dict(self.default_mpi)
            for key in mpi:
                if key not in self.default_mpi:
                    allowed = ', '.join(list(self.default_mpi.keys()))
                    raise TypeError('Unexpected keyword "{0}" in "mpi" '
                                    'dictionary.  Must be one of: {1}'
                                    .format(key, allowed))
            # Put dictionary into python variable
            self.mpi.update(mpi)
            self.__dict__.update(self.mpi)
        else:
            self.mpi = None

    def run(self):
        '''Check Which Running method we r going to use and run it'''
        if self.pbs is not None:
            run = self.run_pbs
        elif self.mpi is not None:
            run = self.run_mpi
        else:
            run = self.run_openmx
        run()

    def run_openmx(self):
        def isRunning(process=None):
            ''' Check mpi is running'''
            return process.poll() is None
        runfile = get_file_name('.dat', self.label)
        outfile = get_file_name('.log', self.label)
        olddir = os.getcwd()
        abs_dir = os.path.join(olddir, self.directory)
        try:
            os.chdir(abs_dir)
            if self.command is None:
                self.command = 'openmx %s > %s'
            command = self.command
            command = command % (runfile, outfile)
            self.prind(command)
            p = subprocess.Popen(command, shell=True, universal_newlines=True)
            self.print_file(file=outfile, running=isRunning, process=p)
        finally:
            os.chdir(olddir)
        self.prind("Calculation Finished")

    def run_mpi(self):
        """
        Run openmx using MPI method. If keyword `mpi` is declared, it will
        run.
        """
        def isRunning(process=None):
            ''' Check mpi is running'''
            return process.poll() is None
        processes = self.processes
        threads = self.threads
        runfile = get_file_name('.dat', self.label)
        outfile = get_file_name('.log', self.label)
        olddir = os.getcwd()
        abs_dir = os.path.join(olddir, self.directory)
        try:
            os.chdir(abs_dir)
            command = self.get_command(processes, threads, runfile, outfile)
            self.prind(command)
            p = subprocess.Popen(command, shell=True, universal_newlines=True)
            self.print_file(file=outfile, running=isRunning, process=p)
        finally:
            os.chdir(olddir)
        self.prind("Calculation Finished")

    def run_pbs(self, prefix='test'):
        """
        Execute the OpenMX using Plane Batch System. In order to use this,
        Your system should have Scheduler. PBS
        Basically, it does qsub. and wait until qstat signal shows c
        Super computer user
        """
        nodes = self.nodes
        processes = self.processes

        prefix = self.prefix
        olddir = os.getcwd()
        try:
            os.chdir(self.abs_directory)
        except AttributeError:
            os.chdir(self.directory)

        def isRunning(jobNum=None, status='Q', qstat='qstat'):
            """
            Check submitted job is still Running
            """
            def runCmd(exe):
                p = subprocess.Popen(exe, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True)
                while True:
                    line = p.stdout.readline()
                    if line != '':
                        # the real code does filtering here
                        yield line.rstrip()
                    else:
                        break
            jobs = runCmd('qstat')
            columns = None
            for line in jobs:
                if str(jobNum) in line:
                    columns = line.split()
                    self.prind(line)
            if columns is not None:
                return columns[-2] == status
            else:
                return False

        inputfile = self.label + '.dat'
        outfile = self.label + '.log'

        bashArgs = "#!/bin/bash \n cd $PBS_O_WORKDIR\n"
        jobName = prefix
        cmd = bashArgs + \
            'mpirun -hostfile $PBS_NODEFILE openmx %s > %s' % (
                inputfile, outfile)
        echoArgs = ["echo", "$' %s'" % cmd]
        qsubArgs = ["qsub", "-N", jobName, "-l", "nodes=%d:ppn=%d" %
                    (nodes, processes), "-l", "walltime=" + self.walltime]
        wholeCmd = " ".join(echoArgs) + " | " + " ".join(qsubArgs)
        self.prind(wholeCmd)
        out = subprocess.Popen(wholeCmd, shell=True,
                               stdout=subprocess.PIPE, universal_newlines=True)
        out = out.communicate()[0]
        jobNum = int(re.match(r'(\d+)', out.split()[0]).group(1))

        self.prind('Queue number is ' + str(jobNum) +
                   '\nWaiting for the Queue to start')
        while isRunning(jobNum, status='Q'):
            time.sleep(5)
            self.prind('.')
        self.prind('Start Calculating')
        self.print_file(file=outfile, running=isRunning,
                        jobNum=jobNum, status='R', qstat='qstat')

        os.chdir(olddir)
        self.prind('Calculation Finished!')
        return jobNum

    def clean(self, prefix='test', queue_num=None):
        """Method which cleans up after a calculation.

        The default files generated OpenMX will be deleted IF this
        method is called.

        """
        self.prind("Cleaning Data")
        fileName = get_file_name('', self.label)
        pbs_Name = get_file_name('', self.label)
        files = [
            # prefix+'.out',#prefix+'.dat',#prefix+'.BAND*',
            fileName + '.cif', fileName + '.dden.cube', fileName + \
            '.ene', fileName + '.md', fileName + '.md2',
            fileName + '.tden.cube', fileName + '.sden.cube', fileName + \
            '.v0.cube', fileName + '.v1.cube',
            fileName + '.vhart.cube', fileName + '.den0.cube', fileName + \
            '.bulk.xyz', fileName + '.den1.cube',
            fileName + '.xyz', pbs_Name + '.o' + \
            str(queue_num), pbs_Name + '.e' + str(queue_num)
        ]
        for f in files:
            try:
                self.prind("Removing" + f)
                os.remove(f)
            except OSError:
                self.prind("There is no such file named " + f)

    def calculate(self, atoms=None, properties=None,
                  system_changes=all_changes):
        """
        Capture the RuntimeError from FileIOCalculator.calculate
        and  add a little debug information from the OpenMX output.
        See base FileIOCalculator for documentation.
        """
        if self.parameters.data_path is None:
            if not 'OPENMX_DFT_DATA_PATH' in os.environ:
                warnings.warn('Please either set OPENMX_DFT_DATA_PATH as an'
                              'enviroment variable or specify dft_data_path as'
                              'a keyword argument')

        self.prind("Start Calculation")
        if properties is None:
            properties = self.implemented_properties
        try:
            Calculator.calculate(self, atoms, properties, system_changes)
            self.write_input(atoms=self.atoms, parameters=self.parameters,
                             properties=properties,
                             system_changes=system_changes)
            self.print_input(debug=self.debug, nohup=self.nohup)
            self.run()
            #  self.read_results()
            self.version = self.read_version()
            output_atoms = read_openmx(filename=self.label, debug=self.debug)
            self.output_atoms = output_atoms
            # XXX The parameters are supposedly inputs, so it is dangerous
            # to update them from the outputs. --askhl
            self.parameters.update(output_atoms.calc.parameters)
            self.results = output_atoms.calc.results
            # self.clean()
        except RuntimeError as e:
            try:
                with open(get_file_name('.log'), 'r') as f:
                    lines = f.readlines()
                debug_lines = 10
                print('##### %d last lines of the OpenMX output' % debug_lines)
                for line in lines[-20:]:
                    print(line.strip())
                print('##### end of openMX output')
                raise e
            except RuntimeError as e:
                raise e

    def write_input(self, atoms=None, parameters=None,
                    properties=[], system_changes=[]):
        """Write input (dat)-file.
        See calculator.py for further details.

        Parameters:
            - atoms        : The Atoms object to write.
            - properties   : The properties which should be calculated.
            - system_changes : List of properties changed since last run.
        """
        # Call base calculator.
        if atoms is None:
            atoms = self.atoms
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        write_openmx(label=self.label, atoms=atoms, parameters=self.parameters,
                     properties=properties, system_changes=system_changes)

    def print_input(self, debug=None, nohup=None):
        """
        For a debugging purpose, print the .dat file
        """
        if debug is None:
            debug = self.debug
        if nohup is None:
            nohup = self.nohup
        self.prind('Reading input file'+self.label)
        filename = get_file_name('.dat', self.label)
        if not nohup:
            with open(filename, 'r') as f:
                while True:
                    line = f.readline()
                    print(line.strip())
                    if not line:
                        break

    def read(self, label):
        self.parameters = {}
        self.set_label(label)
        if label[-5:] in ['.dat', '.out', '.log']:
            label = label[:-4]
        atoms = read_openmx(filename=label, debug=self.debug)
        self.update_atoms(atoms)
        self.parameters.update(atoms.calc.parameters)
        self.results = atoms.calc.results
        self.parameters['restart'] = self.label
        self.parameters['label'] = label

    def read_version(self, label=None):
        version = None
        if label is None:
            label = self.label
        for line in open(get_file_name('.out', label)):
            if line.find('Ver.') != -1:
                version = line.split()[-1]
                break
        return version

    def update_atoms(self, atoms):
        self.atoms = atoms.copy()

    def set(self, **kwargs):
        """Set all parameters.

            Parameters:
                -kwargs  : Dictionary containing the keywords defined in
                           OpenMXParameters.
        """

        for key, value in kwargs.items():
            if key not in self.default_parameters.keys():
                raise KeyError('Unkown keyword "%s" and value "%s".' %
                               (key, value))
            if key == 'xc' and value not in self.default_parameters.allowed_xc:
                raise KeyError('Given xc "%s" is not allowed' % value)
            if key in ['dat_arguments'] and isinstance(value, dict):
                # For values that are dictionaries, verify subkeys, too.
                default_dict = self.default_parameters[key]
                for subkey in kwargs[key]:
                    if subkey not in default_dict:
                        allowed = ', '.join(list(default_dict.keys()))
                        raise TypeError('Unknown subkeyword "{0}" of keyword '
                                        '"{1}".  Must be one of: {2}'
                                        .format(subkey, key, allowed))

        # Find out what parameter has been changed
        changed_parameters = {}
        for key, value in kwargs.items():
            oldvalue = self.parameters.get(key)
            if key not in self.parameters or not equal(value, oldvalue):
                changed_parameters[key] = value
                self.parameters[key] = value

        # Set the parameters
        for key, value in kwargs.items():
            # print(' Setting the %s as %s'%(key, value))
            self.parameters[key] = value

        # If Changed Parameter is Critical, we have to reset the results
        for key, value in changed_parameters.items():
            if key in ['xc', 'kpts', 'energy_cutoff']:
                self.results = {}

        value = kwargs.get('energy_cutoff')
        if value is not None and not (isinstance(value, (float, int))
                                      and value > 0):
            mess = "'%s' must be a positive number(in eV), \
                got '%s'" % ('energy_cutoff', value)
            raise ValueError(mess)

        atoms = kwargs.get('atoms')
        if atoms is not None and self.atoms is None:
            self.atoms = atoms.copy()

    def set_results(self, results):
        # Not Implemented fully
        self.results.update(results)

    def get_command(self, processes, threads, runfile=None, outfile=None):
        # Contruct the command to send to the operating system
        abs_dir = os.getcwd()
        command = ''
        # run processes specified by the system variable OPENMX_COMMAND
        if processes is None:
            command += os.environ.get('OPENMX_COMMAND')
            if command is None:
                warnings.warn('Either specify OPENMX_COMMAND as an environment\
                variable or specify processes as a keyword argument')
        else:  # run with a specified number of processes
            threads_string = ' -nt ' + str(threads)
            if threads is None:
                threads_string = ''
            command += 'mpirun -np ' + \
                str(processes) + ' openmx %s' + threads_string + ' > %s'
        if runfile is None:
            runfile = abs_dir + '/' + self.prefix + '.dat'
        if outfile is None:
            outfile = abs_dir + '/' + self.prefix + '.log'
        try:
            command = command % (runfile, outfile)
            # command += '" > ./%s &' % outfile  # outputs
        except TypeError:  # in case the OPENMX_COMMAND is incompatible
            raise ValueError(
                "The 'OPENMX_COMMAND' environment must " +
                "be a format string" +
                " with four string arguments.\n" +
                "Example : 'mpirun -np 4 openmx ./%s -nt 2 > ./%s'.\n" +
                "Got '%s'" % command)
        return command

    def get_stress(self, atoms=None):
        if atoms is None:
            atoms = self.atoms

        def check_version():
            if LooseVersion(self.version) < '3.8':
                raise PropertyNotImplementedError(
                    'Version lower than 3.8 does not support stress '
                    'calculation.  Your version is %s' % self.version)

        # We may not yet know what version we are, since that can only
        # be seen from the output
        if getattr(self, 'version', None) is not None:
            check_version()

        try:
            stress = self.get_property('stress', atoms)
        except PropertyNotImplementedError:
            # Now we know the version number, either we raise version
            # error or the original error (the latter should not happen)
            check_version()
            raise

        return stress

    def get_band_structure(self, atoms=None, calc=None):
        """
        This is band structure function. It is compatible to
        ase dft module """
        from ase.dft import band_structure
        if type(self['kpts']) is tuple:
            self['kpts'] = self.get_kpoints(band_kpath=self['band_kpath'])
            return band_structure.get_band_structure(self.atoms, self, )

    def get_bz_k_points(self):
        kgrid = self['kpts']
        if type(kgrid) in [int, float]:
            kgrid = kptdensity2monkhorstpack(self.atoms, kgrid, False)
        bz_k_points = []
        n1 = kgrid[0]
        n2 = kgrid[1]
        n3 = kgrid[2]
        for i in range(n1):
            for j in range(n2):
                # Monkhorst Pack Grid [H.J. Monkhorst and J.D. Pack,
                # Phys. Rev. B 13, 5188 (1976)]
                for k in range(n3):
                    bz_k_points.append((0.5 * float(2 * i - n1 + 1) / n1,
                                        0.5 * float(2 * j - n2 + 1) / n2,
                                        0.5 * float(2 * k - n3 + 1) / n3))
        return np.array(bz_k_points)

    def get_ibz_k_points(self):
        if self['band_kpath'] is None:
            return self.get_bz_k_points()
        else:
            return self.get_kpoints(band_kpath=self['band_kpath'])

    def get_kpoints(self, kpts=None, symbols=None, band_kpath=None, eps=1e-5):
        """Convert band_kpath <-> kpts"""
        if kpts is None:
            kpts = []
            band_kpath = np.array(band_kpath)
            band_nkpath = len(band_kpath)
            for i, kpath in enumerate(band_kpath):
                end = False
                nband = int(kpath[0])
                if(band_nkpath == i):
                    end = True
                    nband += 1
                ini = np.array(kpath[1:4], dtype=float)
                fin = np.array(kpath[4:7], dtype=float)
                x = np.linspace(ini[0], fin[0], nband, endpoint=end)
                y = np.linspace(ini[1], fin[1], nband, endpoint=end)
                z = np.linspace(ini[2], fin[2], nband, endpoint=end)
                kpts.extend(np.array([x, y, z]).T)
            return np.array(kpts, dtype=float)
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
            return band_kpath

    def get_lattice_type(self):
        cellpar = cell_to_cellpar(self.atoms.cell)
        abc = cellpar[:3]
        angles = cellpar[3:]
        min_lv = min(abc)
        if abc.ptp() < 0.01 * min_lv:
            if abs(angles - 90).max() < 1:
                return 'cubic'
            elif abs(angles - 60).max() < 1:
                return 'fcc'
            elif abs(angles - np.arccos(-1 / 3.) * 180 / np.pi).max < 1:
                return 'bcc'
        elif abs(angles - 90).max() < 1:
            if abs(abc[0] - abc[1]).min() < 0.01 * min_lv:
                return 'tetragonal'
            else:
                return 'orthorhombic'
        elif abs(abc[0] - abc[1]) < 0.01 * min_lv and \
                abs(angles[2] - 120) < 1 and abs(angles[:2] - 90).max() < 1:
            return 'hexagonal'
        else:
            return 'not special'

    def get_number_of_spins(self):
        try:
            magmoms = self.atoms.get_initial_magnetic_moments()
            if self['scf_spinpolarization'] is None:
                if isinstance(magmoms[0], float):
                    if abs(magmoms).max() < 0.1:
                        return 1
                    else:
                        return 2
                else:
                    raise NotImplementedError
            else:
                if self['scf_spinpolarization'] == 'on':
                    return 2
                elif self['scf_spinpolarization'] == 'nc' or \
                        np.any(self['initial_magnetic_moments_euler_angles']) \
                        is not None:
                    return 1
        except KeyError:
            return 1

    def get_eigenvalues(self, kpt=None, spin=None):
        if self.results.get('eigenvalues') is None:
            self.calculate(self.atoms)
        if kpt is None and spin is None:
            return self.results['eigenvalues']
        else:
            return self.results['eigenvalues'][spin, kpt, :]

    def get_fermi_level(self):
        try:
            fermi_level = self.results['chemical_potential']
        except KeyError:
            self.calculate()
            fermi_level = self.results['chemical_potential']
        return fermi_level

    def get_number_of_bands(self):
        pag = self.parameters.get
        dfd = default_dictionary
        if 'number_of_bands' not in self.results:
            n = 0
            for atom in self.atoms:
                sym = atom.symbol
                orbitals = pag('dft_data_dict', dfd)[sym]['orbitals used']
                d = 1
                for orbital in orbitals:
                    n += d * orbital
                    d += 2
            self.results['number_of_bands'] = n
        return self.results['number_of_bands']

    def dirG(self, dk, bzone=(0, 0, 0)):
        nx, ny, nz = self['wannier_kpts']
        dx = dk // (ny * nz) + bzone[0] * nx
        dy = (dk // nz) % ny + bzone[1] * ny
        dz = dk % nz + bzone[2] * nz
        return dx, dy, dz

    def dk(self, dirG):
        dx, dy, dz = dirG
        nx, ny, nz = self['wannier_kpts']
        return ny * nz * (dx % nx) + nz * (dy % ny) + dz % nz

    def get_wannier_localization_matrix(self, nbands, dirG, nextkpoint=None,
                                        kpoint=None, spin=0, G_I=(0, 0, 0)):
        # only expected to work for no spin polarization
        try:
            self['bloch_overlaps']
        except KeyError:
            self.read_bloch_overlaps()
        dirG = tuple(dirG)
        nx, ny, nz = self['wannier_kpts']
        nr3 = nx * ny * nz
        if kpoint is None and nextkpoint is None:
            return {kpoint: self['bloch_overlaps'
                                 ][kpoint][dirG][:nbands, :nbands
                                                 ] for kpoint in range(nr3)}
        if kpoint is None:
            kpoint = (nextkpoint - self.dk(dirG)) % nr3
        if nextkpoint is None:
            nextkpoint = (kpoint + self.dk(dirG)) % nr3
        if dirG not in self['bloch_overlaps'][kpoint].keys():
            return np.zeros((nbands, nbands), complex)
        return self['bloch_overlaps'][kpoint][dirG][:nbands, :nbands]

    def prind(self, line, debug=None):
        ''' Print the value if debugging mode is on.
            Otherwise, it just ignored'''
        if debug is None:
            debug = self.debug
        if debug:
            print(line)

    def print_file(self, file=None, running=None, **args):
        ''' Print the file while calculation is running'''
        prev_position = 0
        last_position = 0
        while not os.path.isfile(file):
            self.prind('Waiting for %s to come out' % file)
            time.sleep(5)
        with open(file, 'r') as f:
            while running(**args):
                f.seek(last_position)
                new_data = f.read()
                prev_position = f.tell()
                # self.prind('pos', prev_position != last_position)
                if prev_position != last_position:
                    if not self.nohup:
                        print(new_data)
                    last_position = prev_position
                time.sleep(1)
