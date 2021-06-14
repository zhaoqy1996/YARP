from __future__ import division, print_function

import sys
import os
import pickle
import tempfile
import time
import traceback

import numpy as np

from ase.io import read
from ase.parallel import world
from ase.utils import devnull
from ase.constraints import FixAtoms, UnitCellFilter
from ase.optimize import LBFGS
from ase.io.trajectory import Trajectory
from ase.eos import EquationOfState
from ase.calculators.calculator import get_calculator, names as calcnames
from ase.calculators.calculator import PropertyNotImplementedError
import ase.db as db


class CLICommand:
    """Run calculation with one of ASE's calculators.

    Four types of calculations can be done:

    * single point
    * atomic relaxations
    * unit cell + atomic relaxations
    * equation-of-state

    Examples of the four types of calculations:

        ase run emt h2o.xyz
        ase run emt h2o.xyz -f 0.01
        ase run emt cu.traj -s 0.01
        ase run emt cu.traj -E 5,2.0
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('calculator',
                            help='Name of calculator to use.  '
                            'Must be one of: {}.'
                            .format(', '.join(calcnames)))
        CLICommand.add_more_arguments(parser)

    @staticmethod
    def add_more_arguments(parser):
        add = parser.add_argument
        add('names', nargs='*', help='Read atomic structure from this file.')
        add('-p', '--parameters', default='',
            metavar='key=value,...',
            help='Comma-separated key=value pairs of ' +
            'calculator specific parameters.')
        add('-t', '--tag',
            help='String tag added to filenames.')
        add('--properties', default='efsdMm',
            help='Default value is "efsdMm" meaning calculate energy, ' +
            'forces, stress, dipole moment, total magnetic moment and ' +
            'atomic magnetic moments.')
        add('-f', '--maximum-force', type=float,
            help='Relax internal coordinates.')
        add('--constrain-tags',
            metavar='T1,T2,...',
            help='Constrain atoms with tags T1, T2, ...')
        add('-s', '--maximum-stress', type=float,
            help='Relax unit-cell and internal coordinates.')
        add('-E', '--equation-of-state',
            help='Use "-E 5,2.0" for 5 lattice constants ranging from '
            '-2.0 %% to +2.0 %%.')
        add('--eos-type', default='sjeos', help='Selects the type of eos.')
        add('--modify', metavar='...',
            help='Modify atoms with Python statement.  ' +
            'Example: --modify="atoms.positions[-1,2]+=0.1".')
        add('--after', help='Perform operation after calculation.  ' +
            'Example: --after="atoms.calc.write(...)"')
        add('-i', '--interactive', action='store_true')
        add('-c', '--collection')
        add('-d', '--database',
            help='Use a filename with a ".db" extension for a sqlite3 ' +
            'database or a ".json" extension for a simple json database.  ' +
            'Default is no database')
        add('-S', '--skip', action='store_true',
            help='Skip calculations already done.')

    @staticmethod
    def run(args):
        runner = Runner()
        runner.parse(args)
        if runner.errors:
            sys.exit(runner.errors)


interactive_script = """
import os
import pickle
if "PYTHONSTARTUP" in os.environ:
    exec(open(os.environ["PYTHONSTARTUP"]).read())
from ase.cli.run import Runner
args = pickle.loads({!r})
atoms = Runner().parse(args, True)
"""


class Runner:
    def __init__(self):
        self.db = None
        self.args = None
        self.errors = 0
        self.names = []
        self.calculator_name = None

        if world.rank == 0:
            self.logfile = sys.stdout
        else:
            self.logfile = devnull

    def parse(self, args, interactive=False):
        if not interactive and args.interactive:
            fd = tempfile.NamedTemporaryFile('w')
            fd.write(interactive_script.format(pickle.dumps(args, protocol=0)))
            fd.flush()
            os.system('python3 -i ' + fd.name)
            return

        self.calculator_name = args.calculator

        self.args = args
        atoms = self.run()
        return atoms

    def log(self, *args, **kwargs):
        print(file=self.logfile, *args, **kwargs)

    def run(self):
        args = self.args
        if self.db is None:
            # Create database connection:
            self.db = db.connect(args.database, use_lock_file=True)

        self.expand(args.names)

        if not args.names:
            args.names.insert(0, '-')

        atoms = None
        for name in args.names:
            if atoms is not None:
                del atoms.calc  # release resources from last calculation
            atoms = self.build(name)
            if args.modify:
                exec(args.modify, {'atoms': atoms, 'np': np})

            if name == '-':
                name = atoms.info['key_value_pairs']['name']

            skip = False
            id = None

            if args.skip:
                id = self.db.reserve(name=name)
                if id is None:
                    skip = True

            if not skip:
                self.set_calculator(atoms, name)

                tstart = time.time()
                try:
                    self.log('Running:', name)
                    data = self.calculate(atoms, name)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    self.log(name, 'FAILED')
                    traceback.print_exc(file=self.logfile)
                    tstop = time.time()
                    data = {'time': tstop - tstart}
                    self.errors += 1
                else:
                    tstop = time.time()
                    data['time'] = tstop - tstart
                    self.db.write(atoms, name=name, data=data)

                if id:
                    del self.db[id]

        return atoms

    def calculate(self, atoms, name):
        args = self.args

        data = {}
        if args.maximum_force or args.maximum_stress:
            data = self.optimize(atoms, name)
        if args.equation_of_state:
            data.update(self.eos(atoms, name))
        data.update(self.calculate_once(atoms, name))

        if args.after:
            exec(args.after, {'atoms': atoms, 'data': data})

        return data

    def expand(self, names):
        if not self.names and self.args.collection:
            con = db.connect(self.args.collection)
            self.names = [dct.id for dct in con.select()]
        if not names:
            names[:] = self.names
            return
        if not self.names:
            return
        i = 0
        while i < len(names):
            name = names[i]
            if name.count('-') == 1:
                s1, s2 = name.split('-')
                if s1 in self.names and s2 in self.names:
                    j1 = self.names.index(s1)
                    j2 = self.names.index(s2)
                    names[i:i + 1] = self.names[j1:j2 + 1]
                    i += j2 - j1
            i += 1

    def build(self, name):
        if name == '-':
            con = db.connect(sys.stdin, 'json')
            return con.get_atoms(add_additional_information=True)
        elif self.args.collection:
            con = db.connect(self.args.collection)
            return con.get_atoms(name)
        else:
            atoms = read(name)
            if isinstance(atoms, list):
                assert len(atoms) == 1
                atoms = atoms[0]
            return atoms

    def set_calculator(self, atoms, name):
        cls = get_calculator(self.calculator_name)
        parameters = str2dict(self.args.parameters)
        if getattr(cls, 'nolabel', False):
            atoms.calc = cls(**parameters)
        else:
            atoms.calc = cls(label=self.get_filename(name), **parameters)

    def calculate_once(self, atoms, name):
        args = self.args

        for p in args.properties or 'efsdMm':
            property, method = {'e': ('energy', 'get_potential_energy'),
                                'f': ('forces', 'get_forces'),
                                's': ('stress', 'get_stress'),
                                'd': ('dipole', 'get_dipole_moment'),
                                'M': ('magmom', 'get_magnetic_moment'),
                                'm': ('magmoms', 'get_magnetic_moments')}[p]
            try:
                getattr(atoms, method)()
            except PropertyNotImplementedError:
                pass

        data = {}

        return data

    def optimize(self, atoms, name):
        args = self.args
        if args.constrain_tags:
            tags = [int(t) for t in args.constrain_tags.split(',')]
            mask = [t in tags for t in atoms.get_tags()]
            atoms.constraints = FixAtoms(mask=mask)

        trajectory = Trajectory(self.get_filename(name, 'traj'), 'w', atoms)
        if args.maximum_stress:
            optimizer = LBFGS(UnitCellFilter(atoms), logfile=self.logfile)
            fmax = args.maximum_stress
        else:
            optimizer = LBFGS(atoms, logfile=self.logfile)
            fmax = args.maximum_force

        optimizer.attach(trajectory)
        optimizer.run(fmax=fmax)

        data = {}
        if hasattr(optimizer, 'force_calls'):
            data['force_calls'] = optimizer.force_calls

        return data

    def eos(self, atoms, name):
        args = self.args

        traj = Trajectory(self.get_filename(name, 'traj'), 'w', atoms)

        N, eps = args.equation_of_state.split(',')
        N = int(N)
        eps = float(eps) / 100
        strains = np.linspace(1 - eps, 1 + eps, N)
        v1 = atoms.get_volume()
        volumes = strains**3 * v1
        energies = []
        cell1 = atoms.cell
        for s in strains:
            atoms.set_cell(cell1 * s, scale_atoms=True)
            energies.append(atoms.get_potential_energy())
            traj.write(atoms)
        traj.close()
        eos = EquationOfState(volumes, energies, args.eos_type)
        v0, e0, B = eos.fit()
        atoms.set_cell(cell1 * (v0 / v1)**(1 / 3), scale_atoms=True)
        data = {'volumes': volumes,
                'energies': energies,
                'fitted_energy': e0,
                'fitted_volume': v0,
                'bulk_modulus': B,
                'eos_type': args.eos_type}
        return data

    def get_filename(self, name=None, ext=None):
        if name is None:
            if self.args.tag is None:
                filename = 'ase'
            else:
                filename = self.args.tag
        else:
            if '.' in name:
                name = name.rsplit('.', 1)[0]
            if self.args.tag is None:
                filename = name
            else:
                filename = name + '-' + self.args.tag

        if ext:
            filename += '.' + ext

        return filename


def str2dict(s, namespace={}, sep='='):
    """Convert comma-separated key=value string to dictionary.

    Examples:

    >>> str2dict('xc=PBE,nbands=200,parallel={band:4}')
    {'xc': 'PBE', 'nbands': 200, 'parallel': {'band': 4}}
    >>> str2dict('a=1.2,b=True,c=ab,d=1,2,3,e={f:42,g:cd}')
    {'a': 1.2, 'c': 'ab', 'b': True, 'e': {'g': 'cd', 'f': 42}, 'd': (1, 2, 3)}
    """

    def myeval(value):
        try:
            value = eval(value, namespace)
        except (NameError, SyntaxError):
            pass
        return value

    dct = {}
    s = (s + ',').split(sep)
    for i in range(len(s) - 1):
        key = s[i]
        m = s[i + 1].rfind(',')
        value = s[i + 1][:m]
        if value[0] == '{':
            assert value[-1] == '}'
            value = str2dict(value[1:-1], namespace, ':')
        elif value[0] == '(':
            assert value[-1] == ')'
            value = [myeval(t) for t in value[1:-1].split(',')]
        else:
            value = myeval(value)
        dct[key] = value
        s[i + 1] = s[i + 1][m + 1:]
    return dct
