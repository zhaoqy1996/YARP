"""
This module defines an ASE interface to Turbomole: http://www.turbomole.com/

QMMM functionality provided by Markus Kaukonen <markus.kaukonen@iki.fi>.

Please read the license file (../../LICENSE)

Contact: Ivan Kondov <ivan.kondov@kit.edu>
"""
from __future__ import print_function
import os
import re
import warnings
from subprocess import Popen, PIPE
from math import log10, floor
import numpy as np
from ase import Atoms
from ase.units import Ha, Bohr
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import PropertyNotImplementedError, ReadError
from ase.utils import basestring


def read_output(regex):
    """collects all matching strings from the output"""
    hitlist = []
    checkfiles = []
    for filename in os.listdir('.'):
        if filename.startswith('job.') or filename.endswith('.out'):
            checkfiles.append(filename)
    for filename in checkfiles:
        with open(filename, 'rt') as f:
            lines = f.readlines()
            for line in lines:
                match = re.search(regex, line)
                if match:
                    hitlist.append(match.group(1))
    return hitlist


def execute(args, input_str=None, error_test=True,
            stdout_tofile=True):
    """executes a turbomole executable and process the outputs"""

    if isinstance(args, basestring):
        args = args.split()

    if stdout_tofile:
        stdout_file = 'ASE.TM.' + args[0] + '.out'
        stdout = open(stdout_file, 'w')
    else:
        stdout = PIPE

    if input_str:
        stdin = input_str.encode()
    else:
        stdin = None

    message = 'TM command "' + args[0] + '" execution failed'
    try:
        proc = Popen(args, stdin=PIPE, stderr=PIPE, stdout=stdout)
        res = proc.communicate(input=stdin)
        if error_test:
            error = res[1].decode()
            if 'abnormally' in error or 'ended normally' not in error:
                message += ' with error:\n' + error
                message += '\nSee file ' + stdout_file + ' for details.\n'
                raise RuntimeError(message)
    except RuntimeError as err:
        raise err
    except OSError as err:
        raise OSError(err.args[1] + '\n' + message)
    else:
        print('TM command: "' + args[0] + '" successfully executed')

    if not stdout_tofile:
        return res[0].decode()


def add_data_group(data_group, string=None, raw=False):
    """write a turbomole data group to control file"""
    if raw:
        data = data_group
    else:
        data = '$' + data_group
        if string:
            data += ' ' + string
        data += '\n'
    f = open('control', 'r+')
    lines = f.readlines()
    f.seek(0)
    f.truncate()
    lines.insert(2, data)
    f.write(''.join(lines))
    f.close()


def read_data_group(data_group):
    """read a turbomole data group from control file"""
    args = ['sdg', data_group]
    dg = execute(args, error_test=False, stdout_tofile=False)
    return dg.strip()


def delete_data_group(data_group):
    """delete a turbomole data group from control file"""
    command = ['kdg', data_group]
    execute(command, error_test=False, stdout_tofile=False)


class TurbomoleOptimizer:
    def __init__(self, atoms, calc):
        self.atoms = atoms
        self.calc = calc
        self.atoms.calc = self.calc

    def todict(self):
        return {'type': 'optimization',
                'optimizer': 'TurbomoleOptimizer'}

    def run(self, fmax=None, steps=None):
        if fmax is not None:
            self.calc.parameters['force convergence'] = fmax
            self.calc.verify_parameters()
        if steps is not None:
            self.calc.parameters['geometry optimization iterations'] = steps
            self.calc.verify_parameters()
        self.calc.calculate()
        self.atoms.positions[:] = self.calc.atoms.positions
        self.calc.parameters['task'] = 'energy'


class Turbomole(FileIOCalculator):

    """constants"""
    name = 'Turbomole'

    implemented_properties = ['energy', 'forces', 'dipole', 'free_energy',
                              'charges']

    available_functionals = [
        'slater-dirac-exchange', 's-vwn', 'vwn', 's-vwn_Gaussian', 'pwlda',
        'becke-exchange', 'b-lyp', 'b-vwn', 'lyp', 'b-p', 'pbe', 'tpss',
        'bh-lyp', 'b3-lyp', 'b3-lyp_Gaussian', 'pbe0', 'tpssh', 'lhf', 'oep',
        'b97-d', 'b2-plyp'
    ]
    tm_files = [
        'control', 'coord', 'basis', 'auxbasis', 'energy', 'gradient', 'mos',
        'alpha', 'beta', 'statistics', 'GEO_OPT_CONVERGED', 'GEO_OPT_FAILED',
        'not.converged', 'nextstep', 'hessapprox', 'job.last', 'job.start',
        'optinfo', 'statistics', 'converged', 'vibspectrum',
        'vib_normal_modes', 'hessian', 'dipgrad', 'dscf_problem', 'pc.txt',
        'pc_gradients.txt'
    ]
    tm_tmp_files = [
        'errvec', 'fock', 'oldfock', 'dens', 'ddens', 'diff_densmat',
        'diff_dft_density', 'diff_dft_oper', 'diff_fockmat', 'diis_errvec',
        'diis_oldfock'
    ]
    spec_names = {
        'default': 'default_parameters',
        'comment': 'parameter_comment',
        'updateable': 'parameter_updateable',
        'type': 'parameter_type',
        'key': 'parameter_key',
        'group': 'parameter_group',
        'units': 'parameter_units',
        'mapping': 'parameter_mapping',
        'non-define': 'parameter_no_define'
    }

    # flat dictionaries with parameters attributes
    default_parameters = {}
    parameter_comment = {}
    parameter_updateable = {}
    parameter_type = {}
    parameter_key = {}
    parameter_group = {}
    parameter_units = {}
    parameter_mapping = {}
    parameter_no_define = {}

    # nested dictionary with parameters attributes
    parameter_spec = {
        'automatic orbital shift': {
            'comment': None,
            'default': 0.1,
            'group': 'scforbitalshift',
            'key': 'automatic',
            'mapping': {
                'to_control': lambda a: a / Ha,
                'from_control': lambda a: a * Ha
            },
            'type': float,
            'units': 'eV',
            'updateable': True
        },
        'basis set definition': {
            'comment': 'used only in restart',
            'default': None,
            'group': 'basis',
            'key': None,
            'type': dict,
            'units': None,
            'updateable': False
        },
        'basis set name': {
            'comment': 'current default from module "define"',
            'default': 'def-SV(P)',
            'group': 'basis',
            'key': None,
            'type': str,
            'units': None,
            'updateable': False
        },
        'closed-shell orbital shift': {
            'comment': 'does not work with automatic',
            'default': None,
            'group': 'scforbitalshift',
            'key': 'closedshell',
            'mapping': {
                'to_control': lambda a: a / Ha,
                'from_control': lambda a: a * Ha
            },
            'type': float,
            'units': 'eV',
            'updateable': True
        },
        'damping adjustment step': {
            'comment': None,
            'default': None,
            'group': 'scfdamp',
            'key': 'step',
            'type': float,
            'units': None,
            'updateable': True
        },
        'density convergence': {
            'comment': None,
            'default': None,
            'group': 'denconv',
            'key': 'denconv',
            'mapping': {
                'to_control': lambda a: int(-log10(a)),
                'from_control': lambda a: 10**(-a)
            },
            'non-define': True,
            'type': float,
            'units': None,
            'updateable': True
        },
        'density functional': {
            'comment': None,
            'default': 'b-p',
            'group': 'dft',
            'key': 'functional',
            'type': str,
            'units': None,
            'updateable': True
        },
        'energy convergence': {
            'comment': 'jobex -energy <int>',
            'default': None,
            'group': None,
            'key': None,
            'mapping': {
                'to_control': lambda a: a / Ha,
                'from_control': lambda a: a * Ha
            },
            'type': float,
            'units': 'eV',
            'updateable': True
        },
        'fermi annealing factor': {
            'comment': None,
            'default': 0.95,
            'group': 'fermi',
            'key': 'tmfac',
            'type': float,
            'units': None,
            'updateable': True
        },
        'fermi final temperature': {
            'comment': None,
            'default': 300,
            'group': 'fermi',
            'key': 'tmend',
            'type': float,
            'units': 'Kelvin',
            'updateable': True
        },
        'fermi homo-lumo gap criterion': {
            'comment': None,
            'default': 0.1,
            'group': 'fermi',
            'key': 'hlcrt',
            'mapping': {
                'to_control': lambda a: a / Ha,
                'from_control': lambda a: a * Ha
            },
            'type': float,
            'units': 'eV',
            'updateable': True
        },
        'fermi initial temperature': {
            'comment': None,
            'default': 300,
            'group': 'fermi',
            'key': 'tmstrt',
            'type': float,
            'units': 'Kelvin',
            'updateable': True
        },
        'fermi stopping criterion': {
            'comment': None,
            'default': 0.001,
            'group': 'fermi',
            'key': 'stop',
            'mapping': {
                'to_control': lambda a: a / Ha,
                'from_control': lambda a: a * Ha
            },
            'type': float,
            'units': 'eV',
            'updateable': True
        },
        'force convergence': {
            'comment': 'jobex -gcart <int>',
            'default': None,
            'group': None,
            'key': None,
            'mapping': {
                'to_control': lambda a: a / Ha * Bohr,
                'from_control': lambda a: a * Ha / Bohr
            },
            'type': float,
            'units': 'eV/Angstrom',
            'updateable': True
        },
        'geometry optimization iterations': {
            'comment': 'jobex -c <int>',
            'default': None,
            'group': None,
            'key': None,
            'type': int,
            'units': None,
            'updateable': True
        },
        'grid size': {
            'comment': None,
            'default': 'm3',
            'group': 'dft',
            'key': 'gridsize',
            'type': str,
            'units': None,
            'updateable': True
        },
        'ground state': {
            'comment': 'only this is currently supported',
            'default': True,
            'group': None,
            'key': None,
            'type': bool,
            'units': None,
            'updateable': False
        },
        'initial damping': {
            'comment': None,
            'default': None,
            'group': 'scfdamp',
            'key': 'start',
            'type': float,
            'units': None,
            'updateable': True
        },
        'initial guess': {
            'comment': '"eht", "hcore" or {"use": "<path/to/control>"}',
            'default': 'eht',
            'group': None,
            'key': None,
            'type': None,
            'units': None,
            'updateable': False
        },
        'minimal damping': {
            'comment': None,
            'default': None,
            'group': 'scfdamp',
            'key': 'min',
            'type': float,
            'units': None,
            'updateable': True
        },
        'multiplicity': {
            'comment': None,
            'default': None,
            'group': None,
            'key': None,
            'type': int,
            'units': None,
            'updateable': False
        },
        'non-automatic orbital shift': {
            'comment': None,
            'default': False,
            'group': 'scforbitalshift',
            'key': 'noautomatic',
            'type': bool,
            'units': None,
            'updateable': True
        },
        'point group': {
            'comment': 'only c1 supported',
            'default': 'c1',
            'group': 'symmetry',
            'key': 'symmetry',
            'type': str,
            'units': None,
            'updateable': False
        },
        'ri memory': {
            'comment': None,
            'default': 1000,
            'group': 'ricore',
            'key': 'ricore',
            'type': int,
            'units': 'Megabyte',
            'updateable': True
        },
        'rohf': {
            'comment': 'used only in restart',
            'default': None,
            'group': None,
            'key': None,
            'type': bool,
            'units': None,
            'updateable': False
        },
        'scf energy convergence': {
            'comment': None,
            'default': None,
            'group': 'scfconv',
            'key': 'scfconv',
            'mapping': {
                'to_control': lambda a: int(floor(-log10(a / Ha))),
                'from_control': lambda a: 10**(-a) * Ha
            },
            'type': float,
            'units': 'eV',
            'updateable': True
        },
        'scf iterations': {
            'comment': None,
            'default': 60,
            'group': 'scfiterlimit',
            'key': 'scfiterlimit',
            'type': int,
            'units': None,
            'updateable': True
        },
        'task': {
            'comment': '"energy calculation" = "energy", '
                       '"gradient calculation" = "gradient", '
                       '"geometry optimization" = "optimize", '
                       '"normal mode analysis" = "frequencies"',
            'default': 'energy',
            'group': None,
            'key': None,
            'type': str,
            'units': None,
            'updateable': True
        },
        'title': {
            'comment': None,
            'default': '',
            'group': 'title',
            'key': 'title',
            'type': str,
            'units': None,
            'updateable': False
        },
        'total charge': {
            'comment': None,
            'default': 0,
            'group': None,
            'key': None,
            'type': int,
            'units': None,
            'updateable': False
        },
        'uhf': {
            'comment': None,
            'default': None,
            'group': 'uhf',
            'key': 'uhf',
            'type': bool,
            'units': None,
            'updateable': False
        },
        'use basis set library': {
            'comment': 'only true implemented',
            'default': True,
            'group': 'basis',
            'key': None,
            'type': bool,
            'units': None,
            'updateable': False
        },
        'use dft': {
            'comment': None,
            'default': True,
            'group': 'dft',
            'key': 'dft',
            'type': bool,
            'units': None,
            'updateable': False
        },
        'use fermi smearing': {
            'comment': None,
            'default': False,
            'group': 'fermi',
            'key': 'fermi',
            'type': bool,
            'units': None,
            'updateable': True
        },
        'use redundant internals': {
            'comment': None,
            'default': False,
            'group': 'redundant',
            'key': None,
            'type': bool,
            'units': None,
            'updateable': False
        },
        'use resolution of identity': {
            'comment': None,
            'default': False,
            'group': 'rij',
            'key': 'rij',
            'type': bool,
            'units': None,
            'updateable': False
        },
        'numerical hessian': {
            'comment': 'NumForce will be used if dictionary exists',
            'default': None,
            'group': None,
            'key': None,
            'type': dict,
            'units': None,
            'updateable': True
        },
        'esp fit': {
            'comment': 'ESP fit',
            'default': None,
            'group': 'esp_fit',
            'key': 'esp_fit',
            'type': str,
            'units': None,
            'updateable': True,
            'non-define': True
        }
    }

    # initialize attributes
    parameters = {}
    results = {}
    initialized = False
    pc_initialized = False
    converged = False
    updated = False
    update_energy = None
    update_forces = None
    update_geometry = None
    update_hessian = None
    atoms = None
    forces = None
    e_total = None
    dipole = None
    charges = None
    version = None
    runtime = None
    datetime = None
    hostname = None
    pcpot = None

    def __init__(self, label=None, calculate_energy='dscf',
                 calculate_forces='grad', post_HF=False, atoms=None,
                 restart=False, define_str=None, control_kdg=None,
                 control_input=None, **kwargs):

        FileIOCalculator.__init__(self)

        self.label = label
        self.calculate_energy = calculate_energy
        self.calculate_forces = calculate_forces
        self.post_HF = post_HF
        self.restart = restart
        self.define_str = define_str
        self.control_kdg = control_kdg
        self.control_input = control_input

        # construct flat dictionaries with parameter attributes
        for p in self.parameter_spec:
            for k in self.spec_names:
                if k in list(self.parameter_spec[p].keys()):
                    subdict = getattr(self, self.spec_names[k])
                    subdict.update({p: self.parameter_spec[p][k]})

        if self.restart:
            self._set_restart(kwargs)
        else:
            self.set_parameters(kwargs)
            self.verify_parameters()
            self.reset()

        if atoms is not None:
            atoms.set_calculator(self)
            self.set_atoms(atoms)

    def __getitem__(self, item):
        return getattr(self, item)

    def _set_restart(self, params_update):
        """constructs atoms, parameters and results from a previous
        calculation"""

        # read results, key parameters and non-key parameters
        self.read_restart()
        params_old = self.read_parameters()

        # filter out non-updateable parameters
        for p in list(params_update.keys()):
            if not self.parameter_updateable[p]:
                del params_update[p]
                warnings.warn('"' + p + '"' + ' cannot be changed')

        # update and verify parameters
        params_new = params_old.copy()
        params_new.update(params_update)
        self.set_parameters(params_new)
        self.verify_parameters()

        # if a define string is specified then run define
        if self.define_str:
            execute('define', input_str=self.define_str)

        # updates data groups in the control file
        if params_update or self.control_kdg or self.control_input:
            self._update_data_groups(params_old, params_update)

        self.initialized = True
        # more precise convergence tests are necessary to set these flags:
        self.update_energy = True
        self.update_forces = True
        self.update_geometry = True
        self.update_hessian = True

    def _update_data_groups(self, params_old, params_update):
        """updates data groups in the control file"""
        # construct a list of data groups to update
        grps = []
        for p in list(params_update.keys()):
            if self.parameter_group[p] is not None:
                grps.append(self.parameter_group[p])

        # construct a dictionary of data groups and update params
        dgs = {}
        for g in grps:
            dgs[g] = {}
            for p in self.parameter_key:
                if g == self.parameter_group[p]:
                    if self.parameter_group[p] == self.parameter_key[p]:
                        if p in list(params_update.keys()):
                            val = params_update[p]
                            pmap = list(self.parameter_mapping.keys())
                            if val is not None and p in pmap:
                                fun = self.parameter_mapping[p]['to_control']
                                val = fun(params_update[p])
                            dgs[g] = val
                    else:
                        if p in list(params_old.keys()):
                            val = params_old[p]
                            pmap = list(self.parameter_mapping.keys())
                            if val is not None and p in pmap:
                                fun = self.parameter_mapping[p]['to_control']
                                val = fun(params_old[p])
                            dgs[g][self.parameter_key[p]] = val
                        if p in list(params_update.keys()):
                            val = params_update[p]
                            pmap = list(self.parameter_mapping.keys())
                            if val is not None and p in pmap:
                                fun = self.parameter_mapping[p]['to_control']
                                val = fun(params_update[p])
                            dgs[g][self.parameter_key[p]] = val

        # write dgs dictionary to a data group
        for g in dgs:
            delete_data_group(g)
            if isinstance(dgs[g], dict):
                string = ''
                for key in list(dgs[g].keys()):
                    if dgs[g][key] is None:
                        continue
                    elif isinstance(dgs[g][key], bool):
                        if dgs[g][key]:
                            string += ' ' + key
                    else:
                        string += ' ' + key + '=' + str(dgs[g][key])
                add_data_group(g, string=string)
            else:
                if isinstance(dgs[g], bool):
                    if dgs[g]:
                        add_data_group(g, string='')
                else:
                    add_data_group(g, string=str(dgs[g]))

        self._set_post_define()

    def _set_post_define(self):
        """non-define keys, user-specified changes in the control file"""
        # process key parameters that are not written with define
        for p in list(self.parameters.keys()):
            if p in list(self.parameter_no_define.keys()):
                if self.parameter_no_define[p]:
                    if self.parameters[p]:
                        if p in list(self.parameter_mapping.keys()):
                            fun = self.parameter_mapping[p]['to_control']
                            val = fun(self.parameters[p])
                        else:
                            val = self.parameters[p]
                        delete_data_group(self.parameter_group[p])
                        add_data_group(self.parameter_group[p], str(val))
                    else:
                        delete_data_group(self.parameter_group[p])

        # delete user-specified data groups
        if self.control_kdg:
            for dg in self.control_kdg:
                delete_data_group(dg)

        # append user-defined input to control
        if self.control_input:
            for inp in self.control_input:
                add_data_group(inp, raw=True)

        # add point charges if pcpot defined:
        if self.pcpot:
            self.set_point_charges()

    def set_parameters(self, params):
        """loads the default parameters and updates with actual values"""
        self.parameters = self.default_parameters.copy()
        self.parameters.update(params)
        if self.parameters['use resolution of identity']:
            self.calculate_energy = 'ridft'
            self.calculate_forces = 'rdgrad'

    def verify_parameters(self):
        """detect wrong or not implemented parameters"""

        # kwargs parameters are ignored if user provides define_str
        if self.define_str is not None:
            assert isinstance(self.define_str, basestring)
            assert len(self.define_str) != 0
            return

        for par in self.parameters:
            assert par in self.parameter_spec, 'invalid parameter: ' + par

        if self.parameters['use dft']:
            func_list = [x.lower() for x in self.available_functionals]
            func = self.parameters['density functional']
            assert func.lower() in func_list, (
                'density functional not available / not supported'
            )

        assert self.parameters['multiplicity'], 'multiplicity not defined'

        if self.parameters['rohf']:
            raise NotImplementedError('ROHF not implemented')
        if self.parameters['initial guess'] not in ['eht', 'hcore']:
            if not (isinstance(self.parameters['initial guess'], dict) and
                    'use' in self.parameters['initial guess'].keys()):
                raise ValueError('Wrong input for initial guess')
        if not self.parameters['use basis set library']:
            raise NotImplementedError('Explicit basis set definition')
        if self.parameters['point group'] != 'c1':
            raise NotImplementedError('Point group not impemeneted')

    def reset(self):
        """removes all turbomole input, output and scratch files,
        and deletes results dict and the atoms object"""
        self.atoms = None
        self.results = {}
        self.results['calculation parameters'] = {}
        ase_files = [f for f in os.listdir('.') if f.startswith('ASE.TM.')]
        for f in self.tm_files + self.tm_tmp_files + ase_files:
            if os.path.exists(f):
                os.remove(f)
        self.initialized = False
        self.pc_initialized = False
        self.converged = False

    def set_atoms(self, atoms):
        """Create the self.atoms object and writes the coord file. If
        self.atoms exists a check for changes and an update of the atoms
        are performed. Note: Only positions changes are tracked in this
        version.
        """
        changes = self.check_state(atoms, tol=1e-13)
        if self.atoms == atoms or 'positions' not in changes:
            # print('two atoms obj are (almost) equal')
            if self.updated and os.path.isfile('coord'):
                self.updated = False
                a = read('coord').get_positions()
                if np.allclose(a, atoms.get_positions(), rtol=0, atol=1e-13):
                    return
            else:
                return

        changes = self.check_state(atoms, tol=1e-2)
        if 'positions' in changes:
            # print(two atoms obj are different')
            self.reset()
        else:
            # print('two atoms obj are slightly different')
            if self.parameters['use redundant internals']:
                self.reset()

        write('coord', atoms)
        self.atoms = atoms.copy()
        self.update_energy = True
        self.update_forces = True
        self.update_geometry = True
        self.update_hessian = True

    def get_define_str(self):
        """construct a define string from the parameters dictionary"""
        define_str_tpl = (
            '\n__title__\na coord\n__inter__\n'
            'bb all __basis_set__\n*\neht\ny\n__charge_str____occ_str__'
            '__single_atom_str____norb_str____dft_str____ri_str__'
            '__scfiterlimit____fermi_str____damp_str__q\n'
        )

        params = self.parameters

        if params['use redundant internals']:
            internals_str = 'ired\n*'
        else:
            internals_str = '*\nno'
        charge_str = str(params['total charge']) + '\n'

        if params['multiplicity'] == 1:
            if params['uhf']:
                occ_str = 'n\ns\n*\n'
            else:
                occ_str = 'y\n'
        elif params['multiplicity'] == 2:
            occ_str = 'y\n'
        elif params['multiplicity'] == 3:
            occ_str = 'n\nt\n*\n'
        else:
            unpaired = params['multiplicity'] - 1
            if params['use fermi smearing']:
                occ_str = 'n\nuf ' + str(unpaired) + '\n*\n'
            else:
                occ_str = 'n\nu ' + str(unpaired) + '\n*\n'

        if len(self.atoms) != 1:
            single_atom_str = ''
        else:
            single_atom_str = '\n'

        if params['multiplicity'] == 1:
            norb_str = ''
        else:
            norb_str = 'n\n'

        if params['use dft']:
            dft_str = 'dft\non\n*\n'
        else:
            dft_str = ''

        if params['density functional']:
            dft_str += 'dft\nfunc ' + params['density functional'] + '\n*\n'

        if params['grid size']:
            dft_str += 'dft\ngrid ' + params['grid size'] + '\n*\n'

        if params['use resolution of identity']:
            ri_str = 'ri\non\nm ' + str(params['ri memory']) + '\n*\n'
        else:
            ri_str = ''

        if params['scf iterations']:
            scfmaxiter = params['scf iterations']
            scfiter_str = 'scf\niter\n' + str(scfmaxiter) + '\n\n'
        else:
            scfiter_str = ''
        if params['scf energy convergence']:
            conv = floor(-log10(params['scf energy convergence'] / Ha))
            scfiter_str += 'scf\nconv\n' + str(int(conv)) + '\n\n'

        fermi_str = ''
        if params['use fermi smearing']:
            fermi_str = 'scf\nfermi\n'
            if params['fermi initial temperature']:
                par = str(params['fermi initial temperature'])
                fermi_str += '1\n' + par + '\n'
            if params['fermi final temperature']:
                par = str(params['fermi final temperature'])
                fermi_str += '2\n' + par + '\n'
            if params['fermi annealing factor']:
                par = str(params['fermi annealing factor'])
                fermi_str += '3\n' + par + '\n'
            if params['fermi homo-lumo gap criterion']:
                par = str(params['fermi homo-lumo gap criterion'])
                fermi_str += '4\n' + par + '\n'
            if params['fermi stopping criterion']:
                par = str(params['fermi stopping criterion'])
                fermi_str += '5\n' + par + '\n'
            fermi_str += '\n\n'

        damp_str = ''
        damp_keys = ('initial damping', 'damping adjustment step',
                     'minimal damping')
        damp_pars = [params[k] for k in damp_keys]
        if any(damp_pars):
            damp_str = 'scf\ndamp\n'
            for par in damp_pars:
                par_str = str(par) if par else ''
                damp_str +=  par_str + '\n'
            damp_str += '\n'

        define_str = define_str_tpl
        define_str = re.sub('__title__', params['title'], define_str)
        define_str = re.sub('__basis_set__', params['basis set name'],
                            define_str)
        define_str = re.sub('__charge_str__', charge_str, define_str)
        define_str = re.sub('__occ_str__', occ_str, define_str)
        define_str = re.sub('__norb_str__', norb_str, define_str)
        define_str = re.sub('__dft_str__', dft_str, define_str)
        define_str = re.sub('__ri_str__', ri_str, define_str)
        define_str = re.sub('__single_atom_str__', single_atom_str,
                            define_str)
        define_str = re.sub('__inter__', internals_str, define_str)
        define_str = re.sub('__scfiterlimit__', scfiter_str, define_str)
        define_str = re.sub('__fermi_str__', fermi_str, define_str)
        define_str = re.sub('__damp_str__', damp_str, define_str)

        return define_str

    def initialize(self):
        """prepare turbomole control file by running module 'define'"""
        if self.initialized:
            return
        self.verify_parameters()
        if not self.atoms:
            raise RuntimeError('atoms missing during initialization')
        if not os.path.isfile('coord'):
            raise IOError('file coord not found')

        if self.define_str is not None:
            define_str = self.define_str
        else:
            define_str = self.get_define_str()

        # run define
        execute('define', input_str=define_str)

        # process non-default initial guess
        iguess = self.parameters['initial guess']
        if isinstance(iguess, dict) and 'use' in iguess.keys():
            # "use" initial guess
            if self.parameters['multiplicity'] != 1 or self.parameters['uhf']:
                define_str = '\n\n\ny\nuse ' + iguess['use'] + '\nn\nn\nq\n'
            else:
                define_str = '\n\n\ny\nuse ' + iguess['use'] + '\nn\nq\n'
            execute('define', input_str=define_str)
        elif self.parameters['initial guess'] == 'hcore':
            # "hcore" initial guess
            if self.parameters['multiplicity'] != 1 or self.parameters['uhf']:
                delete_data_group('uhfmo_alpha')
                delete_data_group('uhfmo_beta')
                add_data_group('uhfmo_alpha', 'none file=alpha')
                add_data_group('uhfmo_beta', 'none file=beta')
            else:
                delete_data_group('scfmo')
                add_data_group('scfmo', 'none file=mos')

        self._set_post_define()

        self.initialized = True
        self.converged = False

    def calculation_required(self, atoms, properties):
        if self.atoms != atoms:
            return True
        for prop in properties:
            if prop == 'energy' and self.e_total is None:
                return True
            elif prop == 'forces' and self.forces is None:
                return True
        return False

    def calculate(self, atoms=None):
        """execute the requested job"""
        if atoms is None:
            atoms = self.atoms
        if self.parameters['task'] in ['energy', 'energy calculation']:
            self.get_potential_energy(atoms)
        if self.parameters['task'] in ['gradient', 'gradient calculation']:
            self.get_forces(atoms)
        if self.parameters['task'] in ['optimize', 'geometry optimization']:
            self.relax_geometry(atoms)
        if self.parameters['task'] in ['frequencies', 'normal mode analysis']:
            self.normal_mode_analysis(atoms)
        self.read_results()

    def relax_geometry(self, atoms=None):
        """execute geometry optimization with script jobex"""
        if atoms is None:
            atoms = self.atoms
        self.set_atoms(atoms)
        if self.converged and not self.update_geometry:
            return
        self.initialize()
        jobex_flags = ''
        if self.parameters['use resolution of identity']:
            jobex_flags += ' -ri'
        if self.parameters['force convergence']:
            par = self.parameters['force convergence']
            conv = floor(-log10(par / Ha * Bohr))
            jobex_flags += ' -gcart ' + str(int(conv))
        if self.parameters['energy convergence']:
            par = self.parameters['energy convergence']
            conv = floor(-log10(par / Ha))
            jobex_flags += ' -energy ' + str(int(conv))
        geom_iter = self.parameters['geometry optimization iterations']
        if geom_iter is not None:
            assert isinstance(geom_iter, int)
            jobex_flags += ' -c ' + str(geom_iter)
        self.converged = False
        execute('jobex' + jobex_flags)
        # check convergence
        self.converged = self.read_convergence()
        if self.converged:
            self.update_energy = False
            self.update_forces = False
            self.update_geometry = False
            self.update_hessian = True
        # read results
        new_struct = read('coord')
        atoms.set_positions(new_struct.get_positions())
        self.atoms = atoms.copy()
        self.read_energy()

    def normal_mode_analysis(self, atoms=None):
        """execute normal mode analysis with modules aoforce or NumForce"""
        from ase.constraints import FixAtoms
        if atoms is None:
            atoms = self.atoms
        self.set_atoms(atoms)
        self.initialize()
        if self.update_energy:
            self.get_potential_energy(atoms)
        if self.update_hessian:
            fixatoms = []
            for constr in atoms.constraints:
                if isinstance(constr, FixAtoms):
                    ckwargs = constr.todict()['kwargs']
                    if 'indices' in ckwargs.keys():
                        fixatoms.extend(ckwargs['indices'])
            if self.parameters['numerical hessian'] is None:
                if len(fixatoms) > 0:
                    define_str = '\n\ny\n'
                    for index in fixatoms:
                        define_str += 'm ' + str(index + 1) + ' 999.99999999\n'
                    define_str += '*\n*\nn\nq\n'
                    execute('define', input_str=define_str)
                    dg = read_data_group('atoms')
                    regex = r'(mass\s*=\s*)999.99999999'
                    dg = re.sub(regex, r'\g<1>9999999999.9', dg)
                    dg += '\n'
                    delete_data_group('atoms')
                    add_data_group(dg, raw=True)
                execute('aoforce')
            else:
                optstr = ''
                pdict = self.parameters['numerical hessian']
                if self.parameters['use resolution of identity']:
                    optstr += ' -ri'
                if len(fixatoms) > 0:
                    optstr += ' -frznuclei -central -c'
                if 'central' in pdict.keys():
                    optstr += ' -central'
                if 'delta' in pdict.keys():
                    optstr += ' -d ' + str(pdict['delta'] / Bohr)
                execute('NumForce' + optstr)
            self.update_hessian = False

    def read_restart(self):
        """read a previous calculation from control file"""
        self.atoms = read('coord')
        self.atoms.set_calculator(self)
        self.converged = self.read_convergence()
        read_methods = [
            self.read_energy,
            self.read_gradient,
            self.read_forces,
            self.read_basis_set,
            self.read_ecps,
            self.read_mos,
            self.read_occupation_numbers,
            self.read_dipole_moment,
            self.read_ssquare,
            self.read_hessian,
            self.read_vibrational_reduced_masses,
            self.read_normal_modes,
            self.read_vibrational_spectrum,
            self.read_charges,
            self.read_point_charges,
            self.read_run_parameters
        ]
        for method in read_methods:
            try:
                method()
            except ReadError as err:
                warnings.warn(err.args[0])

    def read_parameters(self):
        """read parameters from control file"""

        def parse_data_group(dg, dg_name):
            """parse a data group"""
            if len(dg) == 0:
                return None
            lsep = None
            ksep = None
            ndg = dg.replace('$' + dg_name, '').strip()
            if '\n' in ndg:
                lsep = '\n'
            if '=' in ndg:
                ksep = '='
            if not lsep and not ksep:
                return ndg
            result = {}
            lines = ndg.split(lsep)
            for line in lines:
                fields = line.strip().split(ksep)
                if len(fields) == 2:
                    result[fields[0]] = fields[1]
                elif len(fields) == 1:
                    result[fields[0]] = True
            return result

        params = {}
        pdgs = {}
        for p in self.parameter_group:
            if self.parameter_group[p] and self.parameter_key[p]:
                pdgs[p] = parse_data_group(
                    read_data_group(self.parameter_group[p]),
                    self.parameter_group[p]
                )

        for p in self.parameter_key:
            if self.parameter_key[p]:
                if self.parameter_key[p] == self.parameter_group[p]:
                    if pdgs[p] is None:
                        if self.parameter_type[p] is bool:
                            params[p] = False
                        else:
                            params[p] = None
                    else:
                        if self.parameter_type[p] is bool:
                            params[p] = True
                        else:
                            typ = self.parameter_type[p]
                            val = typ(pdgs[p])
                            mapping = self.parameter_mapping
                            if p in list(mapping.keys()):
                                fun = mapping[p]['from_control']
                                val = fun(val)
                            params[p] = val
                else:
                    if pdgs[p] is None:
                        params[p] = None
                    elif isinstance(pdgs[p], basestring):
                        if self.parameter_type[p] is bool:
                            params[p] = (pdgs[p] == self.parameter_key[p])
                    else:
                        if self.parameter_key[p] not in list(pdgs[p].keys()):
                            if self.parameter_type[p] is bool:
                                params[p] = False
                            else:
                                params[p] = None
                        else:
                            typ = self.parameter_type[p]
                            val = typ(pdgs[p][self.parameter_key[p]])
                            mapping = self.parameter_mapping
                            if p in list(mapping.keys()):
                                fun = mapping[p]['from_control']
                                val = fun(val)
                            params[p] = val

        # non-group or non-key parameters

        # per-element and per-atom basis sets not implemented in calculator
        basis_sets = set([bs['nickname'] for bs in self.results['basis set']])
        assert len(basis_sets) == 1
        params['basis set name'] = list(basis_sets)[0]
        params['basis set definition'] = self.results['basis set']

        # rohf, multiplicity and total charge
        orbs = self.results['molecular orbitals']
        params['rohf'] = (bool(len(read_data_group('rohf'))) or
                          bool(len(read_data_group('roothaan'))))
        core_charge = 0
        if self.results['ecps']:
            for ecp in self.results['ecps']:
                for symbol in self.atoms.get_chemical_symbols():
                    if symbol.lower() == ecp['element'].lower():
                        core_charge -= ecp['number of core electrons']
        if params['uhf']:
            alpha_occ = [o['occupancy'] for o in orbs if o['spin'] == 'alpha']
            beta_occ = [o['occupancy'] for o in orbs if o['spin'] == 'beta']
            spin = (np.sum(alpha_occ) - np.sum(beta_occ)) * 0.5
            params['multiplicity'] = int(2 * spin + 1)
            nuclear_charge = np.sum(self.atoms.numbers)
            electron_charge = -int(np.sum(alpha_occ) + np.sum(beta_occ))
            electron_charge += core_charge
            params['total charge'] = nuclear_charge + electron_charge
        elif not params['rohf']:  # restricted HF (closed shell)
            params['multiplicity'] = 1
            nuclear_charge = np.sum(self.atoms.numbers)
            electron_charge = -int(np.sum([o['occupancy'] for o in orbs]))
            electron_charge += core_charge
            params['total charge'] = nuclear_charge + electron_charge
        else:
            raise NotImplementedError('ROHF not implemented')

        # task-related parameters
        if os.path.exists('job.start'):
            with open('job.start', 'r') as log:
                lines = log.readlines()
            for line in lines:
                if 'CRITERION FOR TOTAL SCF-ENERGY' in line:
                    en = int(re.search(r'10\*{2}\(-(\d+)\)', line).group(1))
                    params['energy convergence'] = en
                if 'CRITERION FOR MAXIMUM NORM OF SCF-ENERGY GRADIENT' in line:
                    gr = int(re.search(r'10\*{2}\(-(\d+)\)', line).group(1))
                    params['force convergence'] = gr
                if 'AN OPTIMIZATION WITH MAX' in line:
                    cy = int(re.search(r'MAX. (\d+) CYCLES', line).group(1))
                    params['geometry optimization iterations'] = cy
        return params

    def read_convergence(self):
        """perform convergence checks"""
        if self.restart:
            if bool(len(read_data_group('restart'))):
                return False
            if bool(len(read_data_group('actual'))):
                return False
            if not bool(len(read_data_group('energy'))):
                return False
            if (os.path.exists('job.start') and
                os.path.exists('GEO_OPT_FAILED')):
                return False
            return True

        if self.parameters['task'] in ['optimize', 'geometry optimization']:
            if os.path.exists('GEO_OPT_CONVERGED'):
                return True
            elif os.path.exists('GEO_OPT_FAILED'):
                # check whether a failed scf convergence is the reason
                checkfiles = []
                for filename in os.listdir('.'):
                    if filename.startswith('job.'):
                        checkfiles.append(filename)
                for filename in checkfiles:
                    for line in open(filename):
                        if 'SCF FAILED TO CONVERGE' in line:
                            # scf did not converge in some jobex iteration
                            if filename == 'job.last':
                                raise RuntimeError('scf failed to converge')
                            else:
                                warnings.warn('scf failed to converge')
                warnings.warn('geometry optimization failed to converge')
                return False
            else:
                raise RuntimeError('error during geometry optimization')
        else:
            if os.path.isfile('dscf_problem'):
                raise RuntimeError('scf failed to converge')
            else:
                return True

    def read_results(self):
        """read all results and load them in the results entity"""
        self.read_energy()
        self.read_mos()
        self.read_basis_set()
        self.read_occupation_numbers()
        self.read_dipole_moment()
        self.read_ssquare()
        self.read_run_parameters()
        if self.parameters['task'] in ['gradient', 'optimize',
                                       'gradient calculation',
                                       'geometry optimization']:
            self.read_gradient()
            self.read_forces()
        if self.parameters['task'] in ['frequencies', 'normal mode analysis']:
            self.read_hessian()
            self.read_vibrational_reduced_masses()
            self.read_normal_modes()
            self.read_vibrational_spectrum()
        self.read_charges()

    def read_run_parameters(self):
        """read parameters set by define and not in self.parameters"""

        if 'calculation parameters' not in self.results.keys():
            self.results['calculation parameters'] = {}
        parameters = self.results['calculation parameters']
        dg = read_data_group('symmetry')
        parameters['point group'] = str(dg.split()[1])
        parameters['uhf'] = '$uhf' in read_data_group('uhf')
        # Gaussian function type
        gt = read_data_group('pople')
        if gt == '':
            parameters['gaussian type'] = 'spherical harmonic'
        else:
            gt = gt.split()[1]
            if gt == 'AO':
                parameters['gaussian type'] = 'spherical harmonic'
            elif gt == 'CAO':
                parameters['gaussian type'] = 'cartesian'
            else:
                parameters['gaussian type'] = None

        nvibro = read_data_group('nvibro')
        if nvibro:
            parameters['nuclear degrees of freedom'] = int(nvibro.split()[1])

    def read_energy(self):
        """Read energy from Turbomole energy file."""
        try:
            with open('energy', 'r') as enf:
                text = enf.read().lower()
        except IOError:
            raise ReadError('failed to read energy file')
        if text == '':
            raise ReadError('empty energy file')

        lines = iter(text.split('\n'))

        for line in lines:
            if line.startswith('$end'):
                break
            elif line.startswith('$'):
                pass
            else:
                energy_tmp = float(line.split()[1])
                if self.post_HF:
                    energy_tmp += float(line.split()[4])
        # update energy units
        self.e_total = energy_tmp * Ha
        self.results['total energy'] = self.e_total

    def read_forces(self):
        """Read Forces from Turbomole gradient file."""
        dg = read_data_group('grad')
        if len(dg) == 0:
            return
        file = open('gradient', 'r')
        lines = file.readlines()
        file.close()

        forces = np.array([[0, 0, 0]])

        nline = len(lines)
        iline = -1

        for i in range(nline):
            if 'cycle' in lines[i]:
                iline = i

        if iline < 0:
            raise RuntimeError('Please check TURBOMOLE gradients')

        # next line
        iline += len(self.atoms) + 1
        # $end line
        nline -= 1
        # read gradients
        for i in range(iline, nline):
            line = lines[i].replace('D', 'E')
            tmp = np.array([[float(f) for f in line.split()[0:3]]])
            forces = np.concatenate((forces, tmp))
        # Note the '-' sign for turbomole, to get forces
        self.forces = -np.delete(forces, np.s_[0:1], axis=0) * Ha / Bohr
        self.results['energy gradient'] = (-self.forces).tolist()

    def read_occupation_numbers(self):
        """read occupation numbers with module 'eiger' """
        if 'molecular orbitals' not in self.results.keys():
            return
        mos = self.results['molecular orbitals']
        args = ['eiger', '--all', '--pview']
        output = execute(args, error_test=False, stdout_tofile=False)
        lines = output.split('\n')
        for line in lines:
            regex = (
                r'^\s+(\d+)\.*\s+(\w*)\s+(\d+)\s+(\S+)'
                r'\s+(\d*\.*\d*)\s+([-+]?\d+\.\d*)'
            )
            match = re.search(regex, line)
            if match:
                orb_index = int(match.group(3))
                if match.group(2) == 'a':
                    spin = 'alpha'
                elif match.group(2) == 'b':
                    spin = 'beta'
                else:
                    spin = None
                ar_index = next(
                    index for (index, molecular_orbital) in enumerate(mos)
                    if (molecular_orbital['index'] == orb_index and
                        molecular_orbital['spin'] == spin)
                )
                mos[ar_index]['index by energy'] = int(match.group(1))
                irrep = str(match.group(4))
                mos[ar_index]['irreducible representation'] = irrep
                if match.group(5) != '':
                    mos[ar_index]['occupancy'] = float(match.group(5))
                else:
                    mos[ar_index]['occupancy'] = float(0)

    def read_mos(self):
        """read the molecular orbital coefficients and orbital energies
        from files mos, alpha and beta"""

        self.results['molecular orbitals'] = []
        mos = self.results['molecular orbitals']
        keywords = ['scfmo', 'uhfmo_alpha', 'uhfmo_beta']
        spin = [None, 'alpha', 'beta']

        for index, keyword in enumerate(keywords):
            flen = None
            mo = {}
            orbitals_coefficients_line = []
            mo_string = read_data_group(keyword)
            if mo_string == '':
                continue
            mo_string += '\n$end'
            lines = mo_string.split('\n')
            for line in lines:
                if re.match(r'^\s*#', line):
                    continue
                if 'eigenvalue' in line:
                    if len(orbitals_coefficients_line) != 0:
                        mo['eigenvector'] = orbitals_coefficients_line
                        mos.append(mo)
                        mo = {}
                        orbitals_coefficients_line = []
                    regex = (r'^\s*(\d+)\s+(\S+)\s+'
                             r'eigenvalue=([\+\-\d\.\w]+)\s')
                    match = re.search(regex, line)
                    mo['index'] = int(match.group(1))
                    mo['irreducible representation'] = str(match.group(2))
                    eig = float(re.sub('[dD]', 'E', match.group(3))) * Ha
                    mo['eigenvalue'] = eig
                    mo['spin'] = spin[index]
                    mo['degeneracy'] = 1
                    continue
                if keyword in line:
                    # e.g. format(4d20.14)
                    regex = r'format\(\d+[a-zA-Z](\d+)\.\d+\)'
                    match = re.search(regex, line)
                    if match:
                        flen = int(match.group(1))
                    if ('scfdump' in line or 'expanded' in line or
                        'scfconv' not in line):
                        self.converged = False
                    continue
                if '$end' in line:
                    if len(orbitals_coefficients_line) != 0:
                        mo['eigenvector'] = orbitals_coefficients_line
                        mos.append(mo)
                    break
                sfields = [line[i:i + flen]
                           for i in range(0, len(line), flen)]
                ffields = [float(f.replace('D', 'E').replace('d', 'E'))
                           for f in sfields]
                orbitals_coefficients_line += ffields

    def read_basis_set(self):
        """read the basis set"""
        self.results['basis set'] = []
        self.results['basis set formatted'] = {}
        bsf = read_data_group('basis')
        self.results['basis set formatted']['turbomole'] = bsf
        lines = bsf.split('\n')
        basis_set = {}
        functions = []
        function = {}
        primitives = []
        read_tag = False
        read_data = False
        for line in lines:
            if len(line.strip()) == 0:
                continue
            if '$basis' in line:
                continue
            if '$end' in line:
                break
            if re.match(r'^\s*#', line):
                continue
            if re.match(r'^\s*\*', line):
                if read_tag:
                    read_tag = False
                    read_data = True
                else:
                    if read_data:
                        # end primitives
                        function['primitive functions'] = primitives
                        function['number of primitives'] = len(primitives)
                        primitives = []
                        functions.append(function)
                        function = {}
                        # end contracted
                        basis_set['functions'] = functions
                        functions = []
                        self.results['basis set'].append(basis_set)
                        basis_set = {}
                        read_data = False
                    read_tag = True
                continue
            if read_tag:
                match = re.search(r'^\s*(\w+)\s+(.+)', line)
                if match:
                    basis_set['element'] = match.group(1)
                    basis_set['nickname'] = match.group(2)
                else:
                    raise RuntimeError('error reading basis set')
            else:
                match = re.search(r'^\s+(\d+)\s+(\w+)', line)
                if match:
                    if len(primitives) is not 0:
                        # end primitives
                        function['primitive functions'] = primitives
                        function['number of primitives'] = len(primitives)
                        primitives = []
                        functions.append(function)
                        function = {}
                        # begin contracted
                    function['shell type'] = str(match.group(2))
                    continue
                regex = (
                    r'^\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
                    r'\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
                )
                match = re.search(regex, line)
                if match:
                    exponent = float(match.group(1))
                    coefficient = float(match.group(3))
                    primitives.append(
                        {'exponent': exponent, 'coefficient': coefficient}
                    )

    def read_ecps(self):
        """read the effective core potentials"""
        ecpf = read_data_group('ecp')
        if not bool(len(ecpf)):
            self.results['ecps'] = None
            self.results['ecps formatted'] = None
            return
        self.results['ecps'] = []
        self.results['ecps formatted'] = {}
        self.results['ecps formatted']['turbomole'] = ecpf
        lines = ecpf.split('\n')
        ecp = {}
        groups = []
        group = {}
        terms = []
        read_tag = False
        read_data = False
        for line in lines:
            if len(line.strip()) == 0:
                continue
            if '$ecp' in line:
                continue
            if '$end' in line:
                break
            if re.match(r'^\s*#', line):
                continue
            if re.match(r'^\s*\*', line):
                if read_tag:
                    read_tag = False
                    read_data = True
                else:
                    if read_data:
                        # end terms
                        group['terms'] = terms
                        group['number of terms'] = len(terms)
                        terms = []
                        groups.append(group)
                        group = {}
                        # end group
                        ecp['groups'] = groups
                        groups = []
                        self.results['ecps'].append(ecp)
                        ecp = {}
                        read_data = False
                    read_tag = True
                continue
            if read_tag:
                match = re.search(r'^\s*(\w+)\s+(.+)', line)
                if match:
                    ecp['element'] = match.group(1)
                    ecp['nickname'] = match.group(2)
                else:
                    raise RuntimeError('error reading ecp')
            else:
                regex = r'ncore\s*=\s*(\d+)\s+lmax\s*=\s*(\d+)'
                match = re.search(regex, line)
                if match:
                    ecp['number of core electrons'] = int(match.group(1))
                    ecp['maximum angular momentum number'] = \
                        int(match.group(2))
                    continue
                match = re.search(r'^(\w(\-\w)?)', line)
                if match:
                    if len(terms) is not 0:
                        # end terms
                        group['terms'] = terms
                        group['number of terms'] = len(terms)
                        terms = []
                        groups.append(group)
                        group = {}
                        # begin group
                    group['title'] = str(match.group(1))
                    continue
                regex = (r'^\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+'
                         r'(\d)\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)')
                match = re.search(regex, line)
                if match:
                    terms.append(
                        {
                            'coefficient': float(match.group(1)),
                            'power of r': float(match.group(3)),
                            'exponent': float(match.group(4))
                        }
                    )

    def read_gradient(self):
        """read all information in file 'gradient'"""
        from ase import Atom
        grad_string = read_data_group('grad')
        if len(grad_string) == 0:
            return
#       try to reuse ase:
#       structures = read('gradient', index=':')
        lines = grad_string.split('\n')
        history = []
        image = {}
        gradient = []
        atoms = Atoms()
        (cycle, energy, norm) = (None, None, None)
        for line in lines:
            # cycle lines
            regex = (
                r'^\s*cycle =\s*(\d+)\s+'
                r'SCF energy =\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+'
                r'\|dE\/dxyz\| =\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
            )
            match = re.search(regex, line)
            if match:
                if len(atoms):
                    image['optimization cycle'] = cycle
                    image['total energy'] = energy
                    image['gradient norm'] = norm
                    image['energy gradient'] = gradient
                    history.append(image)
                    image = {}
                    atoms = Atoms()
                    gradient = []
                cycle = int(match.group(1))
                energy = float(match.group(2)) * Ha
                norm = float(match.group(4)) * Ha / Bohr
                continue
            # coordinate lines
            regex = (
                r'^\s*([-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?)'
                r'\s+([-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?)'
                r'\s+([-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?)'
                r'\s+(\w+)'
            )
            match = re.search(regex, line)
            if match:
                x = float(match.group(1)) * Bohr
                y = float(match.group(3)) * Bohr
                z = float(match.group(5)) * Bohr
                symbol = str(match.group(7))
                atoms += Atom(symbol.capitalize(), (x, y, z))
                continue
            # gradient lines
            regex = (
                r'^\s*([-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?)'
                r'\s+([-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?)'
                r'\s+([-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?)'
            )
            match = re.search(regex, line)
            if match:
                gradx = float(match.group(1).replace('D', 'E')) * Ha / Bohr
                grady = float(match.group(3).replace('D', 'E')) * Ha / Bohr
                gradz = float(match.group(5).replace('D', 'E')) * Ha / Bohr
                gradient.append([gradx, grady, gradz])

        image['optimization cycle'] = cycle
        image['total energy'] = energy
        image['gradient norm'] = norm
        image['energy gradient'] = gradient
        history.append(image)
        self.results['geometry optimization history'] = history

    def read_hessian(self, noproj=False):
        """Read in the hessian matrix"""
        self.results['hessian matrix'] = {}
        self.results['hessian matrix']['array'] = []
        self.results['hessian matrix']['units'] = '?'
        self.results['hessian matrix']['projected'] = True
        self.results['hessian matrix']['mass weighted'] = True
        dg = read_data_group('nvibro')
        if len(dg) == 0:
            return
        nvibro = int(dg.split()[1])
        self.results['hessian matrix']['dimension'] = nvibro
        row = []
        key = 'hessian'
        if noproj:
            key = 'npr' + key
            self.results['hessian matrix']['projected'] = False
        lines = read_data_group(key).split('\n')
        for line in lines:
            if key in line:
                continue
            fields = line.split()
            row.extend(fields[2:len(fields)])
            if len(row) == nvibro:
                # check whether it is mass-weighted
                float_row = [float(element) for element in row]
                self.results['hessian matrix']['array'].append(float_row)
                row = []

    def read_normal_modes(self, noproj=False):
        """Read in vibrational normal modes"""
        self.results['normal modes'] = {}
        self.results['normal modes']['array'] = []
        self.results['normal modes']['projected'] = True
        self.results['normal modes']['mass weighted'] = True
        self.results['normal modes']['units'] = '?'
        dg = read_data_group('nvibro')
        if len(dg) == 0:
            return
        nvibro = int(dg.split()[1])
        self.results['normal modes']['dimension'] = nvibro
        row = []
        key = 'vibrational normal modes'
        if noproj:
            key = 'npr' + key
            self.results['normal modes']['projected'] = False
        lines = read_data_group(key).split('\n')
        for line in lines:
            if key in line:
                continue
            if '$end' in line:
                break
            fields = line.split()
            row.extend(fields[2:len(fields)])
            if len(row) == nvibro:
                # check whether it is mass-weighted
                float_row = [float(element) for element in row]
                self.results['normal modes']['array'].append(float_row)
                row = []

    def read_vibrational_reduced_masses(self):
        """Read vibrational reduced masses"""
        self.results['vibrational reduced masses'] = []
        dg = read_data_group('vibrational reduced masses')
        if len(dg) == 0:
            return
        lines = dg.split('\n')
        for line in lines:
            if '$vibrational' in line:
                continue
            if '$end' in line:
                break
            fields = [float(element) for element in line.split()]
            self.results['vibrational reduced masses'].extend(fields)

    def read_vibrational_spectrum(self, noproj=False):
        """Read the vibrational spectrum"""
        self.results['vibrational spectrum'] = []
        key = 'vibrational spectrum'
        if noproj:
            key = 'npr' + key
        lines = read_data_group(key).split('\n')
        for line in lines:
            dictionary = {}
            regex = (
                r'^\s+(\d+)\s+(\S*)\s+([-+]?\d+\.\d*)'
                r'\s+(\d+\.\d*)\s+(\S+)\s+(\S+)'
            )
            match = re.search(regex, line)
            if match:
                dictionary['mode number'] = int(match.group(1))
                dictionary['irreducible representation'] = str(match.group(2))
                dictionary['frequency'] = {
                    'units': 'cm^-1',
                    'value': float(match.group(3))
                }
                dictionary['infrared intensity'] = {
                    'units': 'km/mol',
                    'value': float(match.group(4))
                }

                if match.group(5) == 'YES':
                    dictionary['infrared active'] = True
                elif match.group(5) == 'NO':
                    dictionary['infrared active'] = False
                else:
                    dictionary['infrared active'] = None

                if match.group(6) == 'YES':
                    dictionary['Raman active'] = True
                elif match.group(6) == 'NO':
                    dictionary['Raman active'] = False
                else:
                    dictionary['Raman active'] = None

                self.results['vibrational spectrum'].append(dictionary)

    def read_ssquare(self):
        """Read the expectation value of S^2 operator"""
        s2_string = read_data_group('ssquare from dscf')
        if s2_string == '':
            return
        string = s2_string.split('\n')[1]
        ssquare = float(re.search(r'^\s*(\d+\.*\d*)', string).group(1))
        self.results['ssquare from scf calculation'] = ssquare

    def read_dipole_moment(self):
        """Read the dipole moment"""
        dip_string = read_data_group('dipole')
        if dip_string == '':
            return
        lines = dip_string.split('\n')
        for line in lines:
            regex = (
                r'^\s+x\s+([-+]?\d+\.\d*)\s+y\s+([-+]?\d+\.\d*)'
                r'\s+z\s+([-+]?\d+\.\d*)\s+a\.u\.'
            )
            match = re.search(regex, line)
            if match:
                dip_vec = [float(match.group(c)) for c in range(1, 4)]
            regex = r'^\s+\| dipole \| =\s+(\d+\.*\d*)\s+debye'
            match = re.search(regex, line)
            if match:
                dip_abs_val = float(match.group(1))
        self.results['electric dipole moment'] = {}
        self.results['electric dipole moment']['vector'] = {
            'array': dip_vec,
            'units': 'a.u.'
        }
        self.results['electric dipole moment']['absolute value'] = {
            'value': dip_abs_val,
            'units': 'Debye'
        }
        self.dipole = np.array(dip_vec) * Bohr

    def read_version(self):
        """read the version from the tm output if stored in a file"""
        versions = read_output(r'TURBOMOLE\s+V(\d+\.\d+)\s+')
        if len(set(versions)) > 1:
            warnings.warn('different turbomole versions detected')
            self.version = list(set(versions))
        elif len(versions) == 0:
            warnings.warn('no turbomole version detected')
            self.version = None
        else:
            self.version = versions[0]

    def read_datetime(self):
        """read the datetime of the most recent calculation
        from the tm output if stored in a file
        """
        datetimes = read_output(
            r'(\d{4}-[01]\d-[0-3]\d([T\s][0-2]\d:[0-5]'
            r'\d:[0-5]\d\.\d+)?([+-][0-2]\d:[0-5]\d|Z)?)')
        if len(datetimes) == 0:
            warnings.warn('no turbomole datetime detected')
            self.datetime = None
        else:
            # take the most recent time stamp
            self.datetime = sorted(datetimes, reverse=True)[0]

    def read_runtime(self):
        """read the total runtime of calculations"""
        hits = read_output(r'total wall-time\s+:\s+(\d+.\d+)\s+seconds')
        if len(hits) == 0:
            warnings.warn('no turbomole runtimes detected')
            self.runtime = None
        else:
            self.runtime = np.sum([float(a) for a in hits])

    def read_hostname(self):
        """read the hostname of the computer on which the calc has run"""
        hostnames = read_output(r'hostname is\s+(.+)')
        if len(set(hostnames)) > 1:
            warnings.warn('runs on different hosts detected')
            self.hostname = list(set(hostnames))
        else:
            self.hostname = hostnames[0]

    def get_optimizer(self, atoms, trajectory=None, logfile=None):
        """returns a TurbomoleOptimizer object"""
        self.parameters['task'] = 'optimize'
        self.verify_parameters()
        return TurbomoleOptimizer(atoms, self)

    def get_results(self):
        """returns the results dictionary"""
        return self.results

    def get_potential_energy(self, atoms, force_consistent=True):
        # update atoms
        self.updated = self.e_total is None
        self.set_atoms(atoms)
        self.initialize()
        # if update of energy is necessary
        if self.update_energy:
            # calculate energy
            execute(self.calculate_energy)
            # check convergence
            self.converged = self.read_convergence()
            if not self.converged:
                return None
            # read energy
            self.read_energy()

        self.update_energy = False
        return self.e_total

    def get_forces(self, atoms):
        # update atoms
        self.updated = self.forces is None
        self.set_atoms(atoms)
        # complete energy calculations
        if self.update_energy:
            self.get_potential_energy(atoms)
        # if update of forces is necessary
        if self.update_forces:
            # calculate forces
            execute(self.calculate_forces)
            # read forces
            self.read_forces()

        self.update_forces = False
        return self.forces.copy()

    def get_dipole_moment(self, atoms):
        if self.update_energy:
            self.get_potential_energy(atoms)
        self.read_dipole_moment()
        return self.dipole

    def get_property(self, name, atoms=None, allow_calculation=True):
        """return the value of a property"""

        if name not in self.implemented_properties:
            # an ugly work around; the caller should test the raised error
            # if name in ['magmom', 'magmoms', 'charges', 'stress']:
            # return None
            raise PropertyNotImplementedError(name)

        if atoms is None:
            atoms = self.atoms.copy()

        persist_property = {
            'energy': 'e_total',
            'forces': 'forces',
            'dipole': 'dipole',
            'free_energy': 'e_total',
            'charges': 'charges'
        }
        property_getter = {
            'energy': self.get_potential_energy,
            'forces': self.get_forces,
            'dipole': self.get_dipole_moment,
            'free_energy': self.get_potential_energy,
            'charges': self.get_charges
        }
        getter_args = {
            'energy': [atoms],
            'forces': [atoms],
            'dipole': [atoms],
            'free_energy': [atoms, True],
            'charges': [atoms]
        }

        if allow_calculation:
            result = property_getter[name](*getter_args[name])
        else:
            if hasattr(self, persist_property[name]):
                result = getattr(self, persist_property[name])
            else:
                result = None

        if isinstance(result, np.ndarray):
            result = result.copy()
        return result

    def get_charges(self, atoms):
        """return partial charges on atoms from an ESP fit"""
        if self.charges is None:
            self.calculate(atoms)
            self.read_charges()
        return self.charges

    def read_charges(self):
        """read partial charges on atoms from an ESP fit"""
        epsfit_defined = ('esp fit' in self.parameters and
                          self.parameters['esp fit'] is not None)
        if epsfit_defined or len(read_data_group('esp_fit')) > 0:
            filename = 'ASE.TM.' + self.calculate_energy + '.out'
            with open(filename, 'r') as infile:
                lines = infile.readlines()
            oklines = None
            for n, line in enumerate(lines):
                if 'atom  radius/au   charge' in line:
                    oklines = lines[n + 1:n + len(self.atoms) + 1]
            if oklines is not None:
                qm_charges = [float(line.split()[3]) for line in oklines]
                self.charges = np.array(qm_charges)

    def get_forces_on_point_charges(self):
        """return forces acting on point charges"""
        self.get_forces(self.atoms)
        lines = read_data_group('point_charge_gradients').split('\n')[1:]
        forces = []
        for line in lines:
            linef = line.strip().replace('D', 'E')
            forces.append([float(x) for x in linef.split()])
        # Note the '-' sign for turbomole, to get forces
        return -np.array(forces) * Ha / Bohr

    def set_point_charges(self, pcpot=None):
        """write external point charges to control"""
        if pcpot is not None and pcpot != self.pcpot:
            self.pcpot = pcpot
        if self.pcpot.mmcharges is None or self.pcpot.mmpositions is None:
            raise RuntimeError('external point charges not defined')

        if not self.pc_initialized:
            if len(read_data_group('point_charges')) == 0:
                add_data_group('point_charges', 'file=pc.txt')
            if len(read_data_group('point_charge_gradients')) == 0:
                add_data_group(
                    'point_charge_gradients',
                    'file=pc_gradients.txt'
                )
            drvopt = read_data_group('drvopt')
            if 'point charges' not in drvopt:
                drvopt += '\n   point charges\n'
                delete_data_group('drvopt')
                add_data_group(drvopt, raw=True)
            self.pc_initialized = True

        if self.pcpot.updated:
            with open('pc.txt', 'w') as pcfile:
                pcfile.write('$point_charges nocheck list\n')
                for (x, y, z), charge in zip(
                        self.pcpot.mmpositions, self.pcpot.mmcharges):
                    pcfile.write('%20.14f  %20.14f  %20.14f  %20.14f\n'
                                 % (x / Bohr, y / Bohr, z / Bohr, charge))
                pcfile.write('$end \n')
            self.pcpot.updated = False

    def read_point_charges(self):
        """read point charges from previous calculation"""
        pcs = read_data_group('point_charges')
        if len(pcs) > 0:
            lines = pcs.split('\n')[1:]
            (charges, positions) = ([], [])
            for line in lines:
                columns = [float(col) for col in line.strip().split()]
                positions.append([col * Bohr for col in columns[0:3]])
                charges.append(columns[3])
            self.pcpot = PointChargePotential(charges, positions)

    def embed(self, charges=None, positions=None):
        """embed atoms in an array of point-charges; function used in
            qmmm calculations."""
        self.pcpot = PointChargePotential(charges, positions)
        return self.pcpot


class PointChargePotential:
    """Point-charge potential for Turbomole"""
    def __init__(self, mmcharges, mmpositions=None):
        self.mmcharges = mmcharges
        self.mmpositions = mmpositions
        self.mmforces = None
        self.updated = True

    def set_positions(self, mmpositions):
        """set the positions of point charges"""
        self.mmpositions = mmpositions
        self.updated = True

    def set_charges(self, mmcharges):
        """set the values of point charges"""
        self.mmcharges = mmcharges
        self.updated = True

    def get_forces(self, calc):
        """forces acting on point charges"""
        self.mmforces = calc.get_forces_on_point_charges()
        return self.mmforces
