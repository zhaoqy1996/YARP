"""Checkpointing and restart functionality for scripts using ASE Atoms objects.

Initialize checkpoint object:

CP = Checkpoint('checkpoints.db')

Checkpointed code block in try ... except notation:

try:
    a, C, C_err = CP.load()
except NoCheckpoint:
    C, C_err = fit_elastic_constants(a)
    CP.save(a, C, C_err)

Checkpoint code block, shorthand notation:

C, C_err = CP(fit_elastic_constants)(a)

Example for checkpointing within an iterative loop, e.g. for searching crack
tip position:

try:
    a, converged, tip_x, tip_y = CP.load()
except NoCheckpoint:
    converged = False
    tip_x = tip_x0
    tip_y = tip_y0
while not converged:
    ... do something to find better crack tip position ...
    converged = ...
    CP.flush(a, converged, tip_x, tip_y)

The simplest way to use checkpointing is through the CheckpointCalculator. It
wraps any calculator object and does a checkpoint whenever a calculation
is performed:

    calc = ...
    cp_calc = CheckpointCalculator(calc)
    atoms.set_calculator(cp_calc)
    e = atoms.get_potential_energy() # 1st time, does calc, writes to checkfile
                                     # subsequent runs, reads from checkpoint
"""

import numpy as np

import ase
from ase.db import connect
from ase.calculators.calculator import Calculator


class NoCheckpoint(Exception):
    pass

    
class DevNull:
    def write(str, *args):
        pass

        
class Checkpoint(object):
    _value_prefix = '_values_'

    def __init__(self, db='checkpoints.db', logfile=None):
        self.db = db
        if logfile is None:
            logfile = DevNull()
        self.logfile = logfile

        self.checkpoint_id = [0]
        self.in_checkpointed_region = False

    def __call__(self, func, *args, **kwargs):
        checkpoint_func_name = str(func)
        
        def decorated_func(*args, **kwargs):
            # Get the first ase.Atoms object.
            atoms = None
            for a in args:
                if atoms is None and isinstance(a, ase.Atoms):
                    atoms = a

            try:
                retvals = self.load(atoms=atoms)
            except NoCheckpoint:
                retvals = func(*args, **kwargs)
                if isinstance(retvals, tuple):
                    self.save(*retvals, atoms=atoms,
                              checkpoint_func_name=checkpoint_func_name)
                else:
                    self.save(retvals, atoms=atoms,
                              checkpoint_func_name=checkpoint_func_name)
            return retvals
        return decorated_func

    def _increase_checkpoint_id(self):
        if self.in_checkpointed_region:
            self.checkpoint_id += [1]
        else:
            self.checkpoint_id[-1] += 1
        self.logfile.write('Entered checkpoint region '
                           '{0}.\n'.format(self.checkpoint_id))

        self.in_checkpointed_region = True

    def _decrease_checkpoint_id(self):
        self.logfile.write('Leaving checkpoint region '
                           '{0}.\n'.format(self.checkpoint_id))
        if not self.in_checkpointed_region:
            self.checkpoint_id = self.checkpoint_id[:-1]
            assert len(self.checkpoint_id) >= 1
        self.in_checkpointed_region = False
        assert self.checkpoint_id[-1] >= 1

    def _mangled_checkpoint_id(self):
        """
        Returns a mangled checkpoint id string:
            check_c_1:c_2:c_3:...
        E.g. if checkpoint is nested and id is [3,2,6] it returns:
            'check3:2:6'
        """
        return 'check'+':'.join(str(id) for id in self.checkpoint_id)

    def load(self, atoms=None):
        """
        Retrieve checkpoint data from file. If atoms object is specified, then
        the calculator connected to that object is copied to all returning
        atoms object.

        Returns tuple of values as passed to flush or save during checkpoint
        write.
        """
        self._increase_checkpoint_id()

        retvals = []
        with connect(self.db) as db:
            try:
                dbentry = db.get(checkpoint_id=self._mangled_checkpoint_id())
            except KeyError:
                raise NoCheckpoint

            data = dbentry.data
            atomsi = data['checkpoint_atoms_args_index']
            i = 0
            while (i == atomsi or
                   '{0}{1}'.format(self._value_prefix, i) in data):
                if i == atomsi:
                    newatoms = dbentry.toatoms()
                    if atoms is not None:
                        # Assign calculator
                        newatoms.set_calculator(atoms.get_calculator())
                    retvals += [newatoms]
                else:
                    retvals += [data['{0}{1}'.format(self._value_prefix, i)]]
                i += 1

        self.logfile.write('Successfully restored checkpoint '
                           '{0}.\n'.format(self.checkpoint_id))
        self._decrease_checkpoint_id()
        if len(retvals) == 1:
            return retvals[0]
        else:
            return tuple(retvals)

    def _flush(self, *args, **kwargs):
        data = dict(('{0}{1}'.format(self._value_prefix, i), v)
                    for i, v in enumerate(args))

        try:
            atomsi = [isinstance(v, ase.Atoms) for v in args].index(True)
            atoms = args[atomsi]
            del data['{0}{1}'.format(self._value_prefix, atomsi)]
        except ValueError:
            atomsi = -1
            try:
                atoms = kwargs['atoms']
            except KeyError:
                raise RuntimeError('No atoms object provided in arguments.')

        try:
            del kwargs['atoms']
        except KeyError:
            pass

        data['checkpoint_atoms_args_index'] = atomsi
        data.update(kwargs)

        with connect(self.db) as db:
            try:
                dbentry = db.get(checkpoint_id=self._mangled_checkpoint_id())
                del db[dbentry.id]
            except KeyError:
                pass
            db.write(atoms, checkpoint_id=self._mangled_checkpoint_id(),
                     data=data)

        self.logfile.write('Successfully stored checkpoint '
                           '{0}.\n'.format(self.checkpoint_id))

    def flush(self, *args, **kwargs):
        """
        Store data to a checkpoint without increasing the checkpoint id. This
        is useful to continously update the checkpoint state in an iterative
        loop.
        """
        # If we are flushing from a successfully restored checkpoint, then
        # in_checkpointed_region will be set to False. We need to reset to True
        # because a call to flush indicates that this checkpoint is still
        # active.
        self.in_checkpointed_region = False
        self._flush(*args, **kwargs)

    def save(self, *args, **kwargs):
        """
        Store data to a checkpoint and increase the checkpoint id. This closes
        the checkpoint.
        """
        self._decrease_checkpoint_id()
        self._flush(*args, **kwargs)


def atoms_almost_equal(a, b, tol=1e-9):
    return (np.abs(a.positions - b.positions).max() < tol and
            (a.numbers == b.numbers).all() and
            np.abs(a.cell - b.cell).max() < tol and
            (a.pbc == b.pbc).all())


class CheckpointCalculator(Calculator):
    """
    This wraps any calculator object to checkpoint whenever a calculation
    is performed.

    This is particularily useful for expensive calculators, e.g. DFT and
    allows usage of complex workflows.

    Example usage:

        calc = ...
        cp_calc = CheckpointCalculator(calc)
        atoms.set_calculator(cp_calc)
        e = atoms.get_potential_energy()
        # 1st time, does calc, writes to checkfile
        # subsequent runs, reads from checkpoint file
    """
    implemented_properties = ase.calculators.calculator.all_properties
    default_parameters = {}
    name = 'CheckpointCalculator'

    property_to_method_name = {
        'energy': 'get_potential_energy',
        'energies': 'get_potential_energies',
        'forces': 'get_forces',
        'stress': 'get_stress',
        'stresses': 'get_stresses'}

    def __init__(self, calculator, db='checkpoints.db', logfile=None):
        Calculator.__init__(self)
        self.calculator = calculator
        if logfile is None:
            logfile = DevNull()
        self.checkpoint = Checkpoint(db, logfile)
        self.logfile = logfile

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        try:
            results = self.checkpoint.load(atoms)
            prev_atoms, results = results[0], results[1:]
            try:
                assert atoms_almost_equal(atoms, prev_atoms)
            except AssertionError:
                raise AssertionError('mismatch between current atoms and '
                                     'those read from checkpoint file')
            self.logfile.write('retrieved results for {0} from checkpoint\n'
                               .format(properties))
            # save results in calculator for next time
            if isinstance(self.calculator, Calculator):
                if not hasattr(self.calculator, 'results'):
                    self.calculator.results = {}
                self.calculator.results.update(dict(zip(properties, results)))
        except NoCheckpoint:
            if isinstance(self.calculator, Calculator):
                self.logfile.write('doing calculation of {0} with new-style '
                                   'calculator interface\n'.format(properties))
                self.calculator.calculate(atoms, properties, system_changes)
                results = [self.calculator.results[prop]
                           for prop in properties]
            else:
                self.logfile.write('doing calculation of {0} with old-style '
                                   'calculator interface\n'.format(properties))
                results = []
                for prop in properties:
                    method_name = self.property_to_method_name[prop]
                    method = getattr(self.calculator, method_name)
                    results.append(method(atoms))
            _calculator = atoms.get_calculator()
            try:
                atoms.set_calculator(self.calculator)
                self.checkpoint.save(atoms, *results)
            finally:
                atoms.set_calculator(_calculator)

        self.results = dict(zip(properties, results))
