import os
from warnings import warn
import subprocess
import numpy as np

from ase.calculators.calculator import (Calculator,
                                        FileIOCalculator,
                                        all_changes,
                                        PropertyNotImplementedError)
from ase.units import Bohr, Hartree
from ase.io.xyz import write_xyz
from ase.io.vasp import write_vasp
from ase.parallel import world


class DFTD3(FileIOCalculator):
    """Grimme DFT-D3 calculator"""

    name = 'DFTD3'
    dftd3_implemented_properties = ['energy', 'forces', 'stress']

    damping_methods = ['zero', 'bj', 'zerom', 'bjm']

    default_parameters = {'xc': None,  # PBE if no custom damping parameters
                          'grad': True,  # calculate forces/stress
                          'abc': False,  # ATM 3-body contribution
                          'cutoff': 95 * Bohr,  # Cutoff for 2-body calcs
                          'cnthr': 40 * Bohr,  # Cutoff for 3-body and CN calcs
                          'old': False,  # use old DFT-D2 method instead
                          'damping': 'zero',  # Default to zero-damping
                          'tz': False,  # 'triple zeta' alt. parameters
                          's6': None,  # damping parameters start here
                          'sr6': None,
                          's8': None,
                          'sr8': None,
                          'alpha6': None,
                          'a1': None,
                          'a2': None,
                          'beta': None}

    dftd3_flags = ('grad', 'pbc', 'abc', 'old', 'tz')

    def __init__(self,
                 label='ase_dftd3',  # Label for dftd3 output files
                 command=None,  # Command for running dftd3
                 dft=None,  # DFT calculator
                 atoms=None,
                 comm=world,
                 **kwargs):

        self.dft = None
        FileIOCalculator.__init__(self, restart=None,
                                  ignore_bad_restart_file=False,
                                  label=label,
                                  atoms=atoms,
                                  command=command,
                                  dft=dft,
                                  **kwargs)

        # If the user is running DFTD3 with another DFT calculator, such as
        # GPAW, the DFT portion of the calculation should take much longer.
        # If we only checked for a valid command in self.calculate, the DFT
        # calculation would run before we realize that we don't know how
        # to run dftd3. So, we check here at initialization time, to avoid
        # wasting the user's time.
        if self.command is None:
            raise RuntimeError("Don't know how to run DFTD3! Please "
                               'set the ASE_DFTD3_COMMAND environment '
                               'variable, or explicitly pass the path '
                               'to the dftd3 executable to the D3 calculator!')
        if isinstance(self.command, str):
            self.command = self.command.split()

        self.comm = comm

    def set(self, **kwargs):
        changed_parameters = {}
        # Convert from 'func' keyword to 'xc'. Internally, we only store
        # 'xc', but 'func' is also allowed since it is consistent with the
        # CLI dftd3 interface.
        if kwargs.get('func'):
            if kwargs.get('xc') and kwargs['func'] != kwargs['xc']:
                raise RuntimeError('Both "func" and "xc" were provided! '
                                   'Please provide at most one of these '
                                   'two keywords. The preferred keyword '
                                   'is "xc"; "func" is allowed for '
                                   'consistency with the CLI dftd3 '
                                   'interface.')
            if kwargs['func'] != self.parameters['xc']:
                changed_parameters['xc'] = kwargs['func']
            self.parameters['xc'] = kwargs['func']

        # dftd3 only implements energy, forces, and stresses (for periodic
        # systems). But, if a DFT calculator is attached, and that calculator
        # implements more properties, we will expose those properties too.
        if 'dft' in kwargs:
            dft = kwargs.pop('dft')
            if dft is not self.dft:
                changed_parameters['dft'] = dft
            if dft is None:
                self.implemented_properties = self.dftd3_implemented_properties
            else:
                self.implemented_properties = dft.implemented_properties
            self.dft = dft

        # If the user did not supply an XC functional, but did attach a
        # DFT calculator that has XC set, then we will use that. Note that
        # DFTD3's spelling convention is different from most, so in general
        # you will have to explicitly set XC for both the DFT calculator and
        # for DFTD3 (and DFTD3's will likely be spelled differently...)
        if self.parameters['xc'] is None and self.dft is not None:
            if self.dft.parameters.get('xc'):
                self.parameters['xc'] = self.dft.parameters['xc']

        # Check for unknown arguments. Don't raise an error, just let the
        # user know that we don't understand what they're asking for.
        unknown_kwargs = set(kwargs) - set(self.default_parameters)
        if unknown_kwargs:
            warn('WARNING: Ignoring the following unknown keywords: {}'
                 ''.format(', '.join(unknown_kwargs)))

        changed_parameters.update(FileIOCalculator.set(self, **kwargs))

        # Ensure damping method is valid (zero, bj, zerom, bjm).
        if self.parameters['damping'] is not None:
            self.parameters['damping'] = self.parameters['damping'].lower()
        if self.parameters['damping'] not in self.damping_methods:
            raise ValueError('Unknown damping method {}!'
                             ''.format(self.parameters['damping']))

        # d2 only is valid with 'zero' damping
        elif self.parameters['old'] and self.parameters['damping'] != 'zero':
            raise ValueError('Only zero-damping can be used with the D2 '
                             'dispersion correction method!')

        # If cnthr (cutoff for three-body and CN calculations) is greater
        # than cutoff (cutoff for two-body calculations), then set the former
        # equal to the latter, since that doesn't make any sense.
        if self.parameters['cnthr'] > self.parameters['cutoff']:
            warn('WARNING: CN cutoff value of {cnthr} is larger than '
                 'regular cutoff value of {cutoff}! Reducing CN cutoff '
                 'to {cutoff}.'
                 ''.format(cnthr=self.parameters['cnthr'],
                           cutoff=self.parameters['cutoff']))
            self.parameters['cnthr'] = self.parameters['cutoff']

        # If you only care about the energy, gradient calculations (forces,
        # stresses) can be bypassed. This will greatly speed up calculations
        # in dense 3D-periodic systems with three-body corrections. But, we
        # can no longer say that we implement forces and stresses.
        if not self.parameters['grad']:
            for val in ['forces', 'stress']:
                if val in self.implemented_properties:
                    self.implemented_properties.remove(val)

        # Check to see if we're using custom damping parameters.
        zero_damppars = {'s6', 'sr6', 's8', 'sr8', 'alpha6'}
        bj_damppars = {'s6', 'a1', 's8', 'a2', 'alpha6'}
        zerom_damppars = {'s6', 'sr6', 's8', 'beta', 'alpha6'}
        all_damppars = zero_damppars | bj_damppars | zerom_damppars

        self.custom_damp = False
        damping = self.parameters['damping']
        damppars = set(kwargs) & all_damppars
        if damppars:
            self.custom_damp = True
            if damping == 'zero':
                valid_damppars = zero_damppars
            elif damping in ['bj', 'bjm']:
                valid_damppars = bj_damppars
            elif damping == 'zerom':
                valid_damppars = zerom_damppars

            # If some but not all damping parameters are provided for the
            # selected damping method, raise an error. We don't have "default"
            # values for damping parameters, since those are stored in the
            # dftd3 executable & depend on XC functional.
            missing_damppars = valid_damppars - damppars
            if missing_damppars and missing_damppars != valid_damppars:
                raise ValueError('An incomplete set of custom damping '
                                 'parameters for the {} damping method was '
                                 'provided! Expected: {}; got: {}'
                                 ''.format(damping,
                                           ', '.join(valid_damppars),
                                           ', '.join(damppars)))

            # If a user provides damping parameters that are not used in the
            # selected damping method, let them know that we're ignoring them.
            # If the user accidentally provided the *wrong* set of parameters,
            # (e.g., the BJ parameters when they are using zero damping), then
            # the previous check will raise an error, so we don't need to
            # worry about that here.
            if damppars - valid_damppars:
                warn('WARNING: The following damping parameters are not '
                     'valid for the {} damping method and will be ignored: {}'
                     ''.format(damping,
                               ', '.join(damppars)))

        # The default XC functional is PBE, but this is only set if the user
        # did not provide their own value for xc or any custom damping
        # parameters.
        if self.parameters['xc'] and self.custom_damp:
            warn('WARNING: Custom damping parameters will be used '
                 'instead of those parameterized for {}!'
                 ''.format(self.parameters['xc']))

        if changed_parameters:
            self.results.clear()
        return changed_parameters

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        # We don't call FileIOCalculator.calculate here, because that method
        # calls subprocess.call(..., shell=True), which we don't want to do.
        # So, we reproduce some content from that method here.
        Calculator.calculate(self, atoms, properties, system_changes)

        # If a parameter file exists in the working directory, delete it
        # first. If we need that file, we'll recreate it later.
        localparfile = os.path.join(self.directory, '.dftd3par.local')
        if world.rank == 0 and os.path.isfile(localparfile):
            os.remove(localparfile)

        # Write XYZ or POSCAR file and .dftd3par.local file if we are using
        # custom damping parameters.
        self.write_input(self.atoms, properties, system_changes)
        command = self._generate_command()

        # Finally, call dftd3 and parse results.
        # DFTD3 does not run in parallel
        # so we only need it to run on 1 core
        errorcode = None
        if self.comm.rank == 0:
            with open(self.label + '.out', 'w') as f:
                errorcode = subprocess.call(command,
                                            cwd=self.directory, stdout=f)

        errorcode = self.comm.broadcast(errorcode, 0)

        if errorcode:
            raise RuntimeError('%s returned an error: %d' %
                               (self.name, errorcode))

        self.read_results()

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties=properties,
                                     system_changes=system_changes)
        # dftd3 can either do fully 3D periodic or non-periodic calculations.
        # It cannot do calculations that are only periodic in 1 or 2
        # dimensions. If the atoms object is periodic in only 1 or 2
        # dimensions, then treat it as a fully 3D periodic system, but warn
        # the user.
        pbc = False
        if any(atoms.pbc):
            if not all(atoms.pbc):
                warn('WARNING! dftd3 can only calculate the dispersion energy '
                     'of non-periodic or 3D-periodic systems. We will treat '
                     'this system as 3D-periodic!')
            pbc = True

        if self.comm.rank == 0:
            if pbc:
                fname = os.path.join(self.directory,
                                     '{}.POSCAR'.format(self.label))
                write_vasp(fname, atoms)
            else:
                fname = os.path.join(
                    self.directory, '{}.xyz'.format(self.label))
                write_xyz(fname, atoms, plain=True)

        # Generate custom damping parameters file. This is kind of ugly, but
        # I don't know of a better way of doing this.
        if self.custom_damp:
            damppars = []
            # s6 is always first
            damppars.append(str(float(self.parameters['s6'])))
            # sr6 is the second value for zero{,m} damping, a1 for bj{,m}
            if self.parameters['damping'] in ['zero', 'zerom']:
                damppars.append(str(float(self.parameters['sr6'])))
            elif self.parameters['damping'] in ['bj', 'bjm']:
                damppars.append(str(float(self.parameters['a1'])))
            # s8 is always third
            damppars.append(str(float(self.parameters['s8'])))
            # sr8 is fourth for zero, a2 for bj{,m}, beta for zerom
            if self.parameters['damping'] == 'zero':
                damppars.append(str(float(self.parameters['sr8'])))
            elif self.parameters['damping'] in ['bj', 'bjm']:
                damppars.append(str(float(self.parameters['a2'])))
            elif self.parameters['damping'] == 'zerom':
                damppars.append(str(float(self.parameters['beta'])))
            # alpha6 is always fifth
            damppars.append(str(int(self.parameters['alpha6'])))
            # last is the version number
            if self.parameters['old']:
                damppars.append('2')
            elif self.parameters['damping'] == 'zero':
                damppars.append('3')
            elif self.parameters['damping'] == 'bj':
                damppars.append('4')
            elif self.parameters['damping'] == 'zerom':
                damppars.append('5')
            elif self.parameters['damping'] == 'bjm':
                damppars.append('6')

            damp_fname = os.path.join(self.directory, '.dftd3par.local')
            if self.comm.rank == 0:
                with open(damp_fname, 'w') as f:
                    f.write(' '.join(damppars))

    def read_results(self):
        # parse the energy
        outname = os.path.join(self.directory, self.label + '.out')
        self.results['energy'] = None
        self.results['free_energy'] = None
        if self.comm.rank == 0:
            with open(outname, 'r') as f:
                for line in f:
                    if line.startswith(' program stopped'):
                        if 'functional name unknown' in line:
                            message = 'Unknown DFTD3 functional name "{}". ' \
                                      'Please check the dftd3.f source file ' \
                                      'for the list of known functionals ' \
                                      'and their spelling.' \
                                      ''.format(self.parameters['xc'])
                        else:
                            message = 'dftd3 failed! Please check the {} ' \
                                      'output file and report any errors ' \
                                      'to the ASE developers.' \
                                      ''.format(outname)
                        raise RuntimeError(message)

                    if line.startswith(' Edisp'):
                        e_dftd3 = float(line.split()[-2]) * Hartree
                        self.results['energy'] = e_dftd3
                        self.results['free_energy'] = e_dftd3
                        break
                else:
                    raise RuntimeError('Could not parse energy from dftd3 '
                                       'output, see file {}'.format(outname))

        self.results['energy'] = self.comm.broadcast(self.results['energy'], 0)
        self.results['free_energy'] = self.comm.broadcast(
            self.results['free_energy'], 0)

        # FIXME: Calculator.get_potential_energy() simply inspects
        # self.results for the free energy rather than calling
        # Calculator.get_property('free_energy'). For example, GPAW does
        # not actually present free_energy as an implemented property, even
        # though it does calculate it. So, we are going to add in the DFT
        # free energy to our own results if it is present in the attached
        # calculator. TODO: Fix the Calculator interface!!!
        if self.dft is not None:
            try:
                efree = self.dft.get_potential_energy(
                    force_consistent=True)
                self.results['free_energy'] += efree
            except PropertyNotImplementedError:
                pass

        if self.parameters['grad']:
            # parse the forces
            forces = np.zeros((len(self.atoms), 3))
            forcename = os.path.join(self.directory, 'dftd3_gradient')
            self.results['forces'] = None
            if self.comm.rank == 0:
                with open(forcename, 'r') as f:
                    for i, line in enumerate(f):
                        forces[i] = np.array([float(x) for x in line.split()])
                self.results['forces'] = -forces * Hartree / Bohr
            self.comm.broadcast(self.results['forces'], 0)

            if any(self.atoms.pbc):
                # parse the stress tensor
                stress = np.zeros((3, 3))
                stressname = os.path.join(self.directory, 'dftd3_cellgradient')
                self.results['stress'] = None
                if self.comm.rank == 0:
                    with open(stressname, 'r') as f:
                        for i, line in enumerate(f):
                            for j, x in enumerate(line.split()):
                                stress[i, j] = float(x)

                    stress *= Hartree / Bohr / self.atoms.get_volume()
                    stress = np.dot(stress, self.atoms.cell.T)
                    self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]
                self.comm.broadcast(self.results['stress'], 0)

    def get_property(self, name, atoms=None, allow_calculation=True):
        dft_result = None
        if self.dft is not None:
            dft_result = self.dft.get_property(name, atoms, allow_calculation)

        dftd3_result = FileIOCalculator.get_property(self, name, atoms,
                                                     allow_calculation)

        if dft_result is None and dftd3_result is None:
            return None
        elif dft_result is None:
            return dftd3_result
        elif dftd3_result is None:
            return dft_result
        else:
            return dft_result + dftd3_result

    def _generate_command(self):
        command = self.command

        if any(self.atoms.pbc):
            command.append(self.label + '.POSCAR')
        else:
            command.append(self.label + '.xyz')

        if not self.custom_damp:
            xc = self.parameters.get('xc')
            if xc is None:
                xc = 'pbe'
            command += ['-func', xc.lower()]

        for arg in self.dftd3_flags:
            if self.parameters.get(arg):
                command.append('-' + arg)

        if any(self.atoms.pbc):
            command.append('-pbc')

        command += ['-cnthr', str(self.parameters['cnthr'] / Bohr)]
        command += ['-cutoff', str(self.parameters['cutoff'] / Bohr)]

        if not self.parameters['old']:
            command.append('-' + self.parameters['damping'])

        return command
