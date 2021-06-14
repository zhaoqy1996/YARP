# -*- coding: utf-8 -*-

"""This module defines an ASE interface to CP2K.

http://www.cp2k.org
Author: Ole Schuett <ole.schuett@mat.ethz.ch>
"""

from __future__ import print_function

import os
import os.path
from warnings import warn
from subprocess import Popen, PIPE
import numpy as np
import ase.io
from ase.units import Rydberg
from ase.calculators.calculator import Calculator, all_changes, Parameters


class CP2K(Calculator):
    """ASE-Calculator for CP2K.

    CP2K is a program to perform atomistic and molecular simulations of solid
    state, liquid, molecular, and biological systems. It provides a general
    framework for different methods such as e.g., density functional theory
    (DFT) using a mixed Gaussian and plane waves approach (GPW) and classical
    pair and many-body potentials.

    CP2K is freely available under the GPL license.
    It is written in Fortran 2003 and can be run efficiently in parallel.

    Check http://www.cp2k.org about how to obtain and install CP2K.
    Make sure that you also have the CP2K-shell available, since it is required
    by the CP2K-calulator.

    The CP2K-calculator relies on the CP2K-shell. The CP2K-shell was originally
    designed for interactive sessions. When a calculator object is
    instantiated, it launches a CP2K-shell as a subprocess in the background
    and communications with it through stdin/stdout pipes. This has the
    advantage that the CP2K process is kept alive for the whole lifetime of
    the calculator object, i.e. there is no startup overhead for a sequence
    of energy evaluations. Furthermore, the usage of pipes avoids slow file-
    system I/O. This mechanism even works for MPI-parallelized runs, because
    stdin/stdout of the first rank are forwarded by the MPI-environment to the
    mpiexec-process.

    The command used by the calculator to launch the CP2K-shell is
    ``cp2k_shell``. To run a parallelized simulation use something like this:

    >>> CP2K.command="env OMP_NUM_THREADS=2 mpiexec -np 4 cp2k_shell.psmp"


    Arguments:

    auto_write: bool
        Flag to enable the auto-write mode. If enabled the
        ``write()`` routine is called after every
        calculation, which mimics the behavior of the
        ``FileIOCalculator``. Default is ``False``.
    basis_set: str
        Name of the basis set to be use.
        The default is ``DZVP-MOLOPT-SR-GTH``.
    basis_set_file: str
        Filename of the basis set file.
        Default is ``BASIS_MOLOPT``.
        Set the environment variable $CP2K_DATA_DIR
        to enabled automatic file discovered.
    charge: float
        The total charge of the system.  Default is ``0``.
    command: str
        The command used to launch the CP2K-shell.
        If ``command`` is not passed as an argument to the
        constructor, the class-variable ``CP2K.command``,
        and then the environment variabel
        ``$ASE_CP2K_COMMAND`` are checked.
        Eventually, ``cp2k_shell`` is used as default.
    cutoff: float
        The cutoff of the finest grid level.  Default is ``400 * Rydberg``.
    debug: bool
        Flag to enable debug mode. This will print all
        communication between the CP2K-shell and the
        CP2K-calculator. Default is ``False``.
    force_eval_method: str
        The method CP2K uses to evaluate energies and forces.
        The default is ``Quickstep``, which is CP2K's
        module for electronic structure methods like DFT.
    inp: str
        CP2K input template. If present, the calculator will
        augment the template, e.g. with coordinates, and use
        it to launch CP2K. Hence, this generic mechanism
        gives access to all features of CP2K.
        Note, that most keywords accept ``None`` to disable the generation
        of the corresponding input section.
    max_scf: int
        Maximum number of SCF iteration to be performed for
        one optimization. Default is ``50``.
    poisson_solver: str
        The poisson solver to be used. Currently, the only supported
        values are ``auto`` and ``None``. Default is ``auto``.
    potential_file: str
        Filename of the pseudo-potential file.
        Default is ``POTENTIAL``.
        Set the environment variable $CP2K_DATA_DIR
        to enabled automatic file discovered.
    pseudo_potential: str
        Name of the pseudo-potential to be use.
        Default is ``auto``. This tries to infer the
        potential from the employed XC-functional,
        otherwise it falls back to ``GTH-PBE``.
    stress_tensor: bool
        Indicates whether the analytic stress-tensor should be calculated.
        Default is ``True``.
    uks: bool
        Requests an unrestricted Kohn-Sham calculations.
        This is need for spin-polarized systems, ie. with an
        odd number of electrons. Default is ``False``.
    xc: str
        Name of exchange and correlation functional.
        Accepts all functions supported by CP2K itself or libxc.
        Default is ``LDA``.
    print_level: str
        PRINT_LEVEL of global output.
        Possible options are:
        DEBUG Everything is written out, useful for debugging purposes only 
        HIGH Lots of output 
        LOW Little output 
        MEDIUM Quite some output 
        SILENT Almost no output 
        Default is 'LOW'
        
    """

    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']
    command = None

    default_parameters = dict(
        auto_write=False,
        basis_set='DZVP-MOLOPT-SR-GTH',
        basis_set_file='BASIS_MOLOPT',
        charge=0,
        cutoff=400 * Rydberg,
        force_eval_method="Quickstep",
        inp='',
        max_scf=50,
        potential_file='POTENTIAL',
        pseudo_potential='auto',
        stress_tensor=True,
        uks=False,
        poisson_solver='auto',
        xc='LDA',
        print_level='LOW')

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='cp2k', atoms=None, command=None,
                 debug=False, **kwargs):
        """Construct CP2K-calculator object."""

        self._debug = debug
        self._force_env_id = None
        self._shell = None
        self.label = None
        self.parameters = None
        self.results = None
        self.atoms = None

        # Several places are check to determine self.command
        if command is not None:
            self.command = command
        elif CP2K.command is not None:
            self.command = CP2K.command
        elif 'ASE_CP2K_COMMAND' in os.environ:
            self.command = os.environ['ASE_CP2K_COMMAND']
        else:
            self.command = 'cp2k_shell'  # default

        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)

        self._shell = Cp2kShell(self.command, self._debug)

        if restart is not None:
            try:
                self.read(restart)
            except:
                if ignore_bad_restart_file:
                    self.reset()
                else:
                    raise

    def __del__(self):
        """Release force_env and terminate cp2k_shell child process"""
        if self._shell:
            self._release_force_env()
            del(self._shell)

    def set(self, **kwargs):
        """Set parameters like set(key1=value1, key2=value2, ...)."""
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def write(self, label):
        'Write atoms, parameters and calculated results into restart files.'
        if self._debug:
            print("Writting restart to: ", label)
        self.atoms.write(label + '_restart.traj')
        self.parameters.write(label + '_params.ase')
        open(label + '_results.ase', 'w').write(repr(self.results))

    def read(self, label):
        'Read atoms, parameters and calculated results from restart files.'
        self.atoms = ase.io.read(label + '_restart.traj')
        self.parameters = Parameters.read(label + '_params.ase')
        results_txt = open(label + '_results.ase').read()
        self.results = eval(results_txt, {'array': np.array})

    def calculate(self, atoms=None, properties=None,
                  system_changes=all_changes):
        """Do the calculation."""

        if not properties:
            properties = ['energy']
        Calculator.calculate(self, atoms, properties, system_changes)

        if self._debug:
            print("system_changes:", system_changes)

        if 'numbers' in system_changes:
            self._release_force_env()

        if self._force_env_id is None:
            self._create_force_env()

        # enable eV and Angstrom as units
        self._shell.send('UNITS_EV_A')
        self._shell.expect('* READY')

        n_atoms = len(self.atoms)
        if 'cell' in system_changes:
            cell = self.atoms.get_cell()
            self._shell.send('SET_CELL %d' % self._force_env_id)
            for i in range(3):
                self._shell.send('%.18e %.18e %.18e' % tuple(cell[i, :]))
            self._shell.expect('* READY')

        if 'positions' in system_changes:
            self._shell.send('SET_POS %d' % self._force_env_id)
            self._shell.send('%d' % (3 * n_atoms))
            for pos in self.atoms.get_positions():
                self._shell.send('%.18e %.18e %.18e' % tuple(pos))
            self._shell.send('*END')
            max_change = float(self._shell.recv())
            assert max_change >= 0 # sanity check
            self._shell.expect('* READY')

        self._shell.send('EVAL_EF %d' % self._force_env_id)
        self._shell.expect('* READY')

        self._shell.send('GET_E %d' % self._force_env_id)
        self.results['energy'] = float(self._shell.recv())
        self.results['free_energy'] = self.results['energy']
        self._shell.expect('* READY')

        forces = np.zeros(shape=(n_atoms, 3))
        self._shell.send('GET_F %d' % self._force_env_id)
        nvals = int(self._shell.recv())
        assert nvals == 3 * n_atoms # sanity check
        for i in range(n_atoms):
            line = self._shell.recv()
            forces[i, :] = [float(x) for x in line.split()]
        self._shell.expect('* END')
        self._shell.expect('* READY')
        self.results['forces'] = forces

        self._shell.send('GET_STRESS %d' % self._force_env_id)
        line = self._shell.recv()
        self._shell.expect('* READY')

        stress = np.array([float(x) for x in line.split()]).reshape(3, 3)
        assert np.all(stress == np.transpose(stress))   # should be symmetric
        # Convert 3x3 stress tensor to Voigt form as required by ASE
        stress = np.array([stress[0, 0], stress[1, 1], stress[2, 2],
                           stress[1, 2], stress[0, 2], stress[0, 1]])
        self.results['stress'] = -1.0 * stress  # cp2k uses the opposite sign

        if self.parameters.auto_write:
            self.write(self.label)

    def _create_force_env(self):
        """Instantiates a new force-environment"""
        assert self._force_env_id is None
        label_dir = os.path.dirname(self.label)
        if len(label_dir) > 0 and not os.path.exists(label_dir):
            print('Creating directory: ' + label_dir)
            os.makedirs(label_dir)  # cp2k expects dirs to exist

        inp = self._generate_input()
        inp_fn = self.label + '.inp'
        out_fn = self.label + '.out'
        self._write_file(inp_fn, inp)
        self._shell.send('LOAD %s %s' % (inp_fn, out_fn))
        self._force_env_id = int(self._shell.recv())
        assert self._force_env_id > 0
        self._shell.expect('* READY')

    def _write_file(self, fn, content):
        """Write content to a file"""
        if self._debug:
            print('Writting to file: ' + fn)
            print(content)
        if self._shell.version < 2.0:
            f = open(fn, 'w')
            f.write(content)
            f.close()
        else:
            lines = content.split('\n')
            if self._shell.version < 2.1:
                lines = [l.strip() for l in lines]  # save chars
            self._shell.send('WRITE_FILE')
            self._shell.send(fn)
            self._shell.send('%d' % len(lines))
            for line in lines:
                self._shell.send(line)
            self._shell.send('*END')
            self._shell.expect('* READY')

    def _release_force_env(self):
        """Destroys the current force-environment"""
        if self._force_env_id:
            if self._shell.isready:
                self._shell.send('DESTROY %d' % self._force_env_id)
                self._shell.expect('* READY')
            else:
                msg = "CP2K-shell not ready, could not release force_env."
                warn(msg, RuntimeWarning)
            self._force_env_id = None

    def _generate_input(self):
        """Generates a CP2K input file"""
        p = self.parameters
        root = parse_input(p.inp)
        root.add_keyword('GLOBAL', 'PROJECT ' + self.label)
        if p.print_level:
            root.add_keyword('GLOBAL', 'PRINT_LEVEL ' + p.print_level)
        if p.force_eval_method:
            root.add_keyword('FORCE_EVAL', 'METHOD ' + p.force_eval_method)
        if p.stress_tensor:
            root.add_keyword('FORCE_EVAL', 'STRESS_TENSOR ANALYTICAL')
            root.add_keyword('FORCE_EVAL/PRINT/STRESS_TENSOR',
                             '_SECTION_PARAMETERS_ ON')
        if p.basis_set_file:
            root.add_keyword('FORCE_EVAL/DFT',
                             'BASIS_SET_FILE_NAME ' + p.basis_set_file)
        if p.potential_file:
            root.add_keyword('FORCE_EVAL/DFT',
                             'POTENTIAL_FILE_NAME ' + p.potential_file)
        if p.cutoff:
            root.add_keyword('FORCE_EVAL/DFT/MGRID',
                             'CUTOFF [eV] %.18e' % p.cutoff)
        if p.max_scf:
            root.add_keyword('FORCE_EVAL/DFT/SCF', 'MAX_SCF %d' % p.max_scf)
            root.add_keyword('FORCE_EVAL/DFT/LS_SCF', 'MAX_SCF %d' % p.max_scf)

        if p.xc:
            legacy_libxc = ""
            for functional in p.xc.split():
                functional = functional.replace("LDA", "PADE")  # resolve alias
                xc_sec = root.get_subsection('FORCE_EVAL/DFT/XC/XC_FUNCTIONAL')
                # libxc input section changed over time
                if functional.startswith("XC_") and self._shell.version < 3.0:
                    legacy_libxc += " " + functional # handled later
                elif functional.startswith("XC_"):
                    s = InputSection(name='LIBXC')
                    s.keywords.append('FUNCTIONAL ' + functional)
                    xc_sec.subsections.append(s)
                else:
                    s = InputSection(name=functional.upper())
                    xc_sec.subsections.append(s)
            if legacy_libxc:
                root.add_keyword('FORCE_EVAL/DFT/XC/XC_FUNCTIONAL/LIBXC',
                                 'FUNCTIONAL ' + legacy_libxc)

        if p.uks:
            root.add_keyword('FORCE_EVAL/DFT', 'UNRESTRICTED_KOHN_SHAM ON')

        if p.charge and p.charge != 0:
            root.add_keyword('FORCE_EVAL/DFT', 'CHARGE %d' % p.charge)

        # add Poisson solver if needed
        if p.poisson_solver == 'auto' and not any(self.atoms.get_pbc()):
            root.add_keyword('FORCE_EVAL/DFT/POISSON', 'PERIODIC NONE')
            root.add_keyword('FORCE_EVAL/DFT/POISSON', 'PSOLVER  MT')

        # write coords
        syms = self.atoms.get_chemical_symbols()
        atoms = self.atoms.get_positions()
        for elm, pos in zip(syms, atoms):
            line = '%s %.18e %.18e %.18e' % (elm, pos[0], pos[1], pos[2])
            root.add_keyword('FORCE_EVAL/SUBSYS/COORD', line, unique=False)

        # write cell
        pbc = ''.join([a for a, b in zip('XYZ', self.atoms.get_pbc()) if b])
        if len(pbc) == 0:
            pbc = 'NONE'
        root.add_keyword('FORCE_EVAL/SUBSYS/CELL', 'PERIODIC ' + pbc)
        c = self.atoms.get_cell()
        for i, a in enumerate('ABC'):
            line = '%s %.18e %.18e %.18e' % (a, c[i, 0], c[i, 1], c[i, 2])
            root.add_keyword('FORCE_EVAL/SUBSYS/CELL', line)

        # determine pseudo-potential
        potential = p.pseudo_potential
        if p.pseudo_potential == 'auto':
            if p.xc and p.xc.upper() in ('LDA', 'PADE', 'BP', 'BLYP', 'PBE',):
                potential = 'GTH-' + p.xc.upper()
            else:
                msg = 'No matching pseudo potential found, using GTH-PBE'
                warn(msg, RuntimeWarning)
                potential = 'GTH-PBE'  # fall back

        # write atomic kinds
        subsys = root.get_subsection('FORCE_EVAL/SUBSYS').subsections
        kinds = dict([(s.params, s) for s in subsys if s.name == "KIND"])
        for elem in set(self.atoms.get_chemical_symbols()):
            if elem not in kinds.keys():
                s = InputSection(name='KIND', params=elem)
                subsys.append(s)
                kinds[elem] = s
            if p.basis_set:
                kinds[elem].keywords.append('BASIS_SET ' + p.basis_set)
            if potential:
                kinds[elem].keywords.append('POTENTIAL ' + potential)

        output_lines = ['!!! Generated by ASE !!!'] + root.write()
        return '\n'.join(output_lines)


class Cp2kShell(object):
    """Wrapper for CP2K-shell child-process"""

    def __init__(self, command, debug):
        """Construct CP2K-shell object"""

        self.isready = False
        self.version = 1.0  # assume oldest possible version until verified
        self._child = None
        self._debug = debug

        # launch cp2k_shell child process
        assert 'cp2k_shell' in command
        if self._debug:
            print(command)
        self._child = Popen(command, shell=True, universal_newlines=True,
                            stdin=PIPE, stdout=PIPE, bufsize=1)
        self.expect('* READY')

        # check version of shell
        self.send('VERSION')
        line = self.recv()
        if not line.startswith('CP2K Shell Version:'):
            raise RuntimeError('Cannot determine version of CP2K shell.  '
                               'Probably the shell version is too old.  '
                               'Please update to CP2K 3.0 or newer.')

        shell_version = line.rsplit(":", 1)[1]
        self.version = float(shell_version)
        assert self.version >= 1.0

        self.expect('* READY')

        # enable harsh mode, stops on any error
        self.send('HARSH')
        self.expect('* READY')

    def __del__(self):
        """Terminate cp2k_shell child process"""
        if self.isready:
            self.send('EXIT')
            rtncode = self._child.wait()
            assert rtncode == 0  # child process exited properly?
        else:
            warn("CP2K-shell not ready, sending SIGTERM.", RuntimeWarning)
            self._child.terminate()
        self._child = None
        self.version = None
        self.isready = False

    def send(self, line):
        """Send a line to the cp2k_shell"""
        assert self._child.poll() is None  # child process still alive?
        if self._debug:
            print('Sending: ' + line)
        if self.version < 2.1 and len(line) >= 80:
            raise Exception('Buffer overflow, upgrade CP2K to r16779 or later')
        assert(len(line) < 800)  # new input buffer size
        self.isready = False
        self._child.stdin.write(line + '\n')

    def recv(self):
        """Receive a line from the cp2k_shell"""
        assert self._child.poll() is None  # child process still alive?
        line = self._child.stdout.readline().strip()
        if self._debug:
            print('Received: ' + line)
        self.isready = line == '* READY'
        return line

    def expect(self, line):
        """Receive a line and asserts that it matches the expected one"""
        received = self.recv()
        assert received == line

class InputSection(object):
    """Represents a section of a CP2K input file"""
    def __init__(self, name, params=None):
        self.name = name.upper()
        self.params = params
        self.keywords = []
        self.subsections = []

    def write(self):
        """Outputs input section as string"""
        output = []
        for k in self.keywords:
            output.append(k)
        for s in self.subsections:
            if s.params:
                output.append('&%s %s' % (s.name, s.params))
            else:
                output.append('&%s' % s.name)
            for l in s.write():
                output.append('   %s' % l)
            output.append('&END %s' % s.name)
        return output

    def add_keyword(self, path, line, unique=True):
        """Adds a keyword to section."""
        parts = path.upper().split('/', 1)
        candidates = [s for s in self.subsections if s.name == parts[0]]
        if len(candidates) == 0:
            s = InputSection(name=parts[0])
            self.subsections.append(s)
            candidates = [s]
        elif len(candidates) != 1:
            raise Exception('Multiple %s sections found ' % parts[0])

        key = line.split()[0].upper()
        if len(parts) > 1:
            candidates[0].add_keyword(parts[1], line, unique)
        elif key == '_SECTION_PARAMETERS_':
            if candidates[0].params is not None:
                msg = 'Section parameter of section %s already set' % parts[0]
                raise Exception(msg)
            candidates[0].params = line.split(' ', 1)[1].strip()
        else:
            old_keys = [k.split()[0].upper() for k in candidates[0].keywords]
            if unique and key in old_keys:
                msg = 'Keyword %s already present in section %s'
                raise Exception(msg % (key, parts[0]))
            candidates[0].keywords.append(line)

    def get_subsection(self, path):
        """Finds a subsection"""
        parts = path.upper().split('/', 1)
        candidates = [s for s in self.subsections if s.name == parts[0]]
        if len(candidates) > 1:
            raise Exception('Multiple %s sections found ' % parts[0])
        if len(candidates) == 0:
            s = InputSection(name=parts[0])
            self.subsections.append(s)
            candidates = [s]
        if len(parts) == 1:
            return candidates[0]
        return candidates[0].get_subsection(parts[1])


def parse_input(inp):
    """Parses the given CP2K input string"""
    root_section = InputSection('CP2K_INPUT')
    section_stack = [root_section]

    for line in inp.split('\n'):
        line = line.split('!', 1)[0].strip()
        if len(line) == 0:
            continue

        if line.upper().startswith('&END'):
            s = section_stack.pop()
        elif line[0] == '&':
            parts = line.split(' ', 1)
            name = parts[0][1:]
            if len(parts) > 1:
                s = InputSection(name=name, params=parts[1].strip())
            else:
                s = InputSection(name=name)
            section_stack[-1].subsections.append(s)
            section_stack.append(s)
        else:
            section_stack[-1].keywords.append(line)

    return root_section
