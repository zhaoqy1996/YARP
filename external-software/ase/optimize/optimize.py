"""Structure optimization. """

import sys
import pickle
import time
from math import sqrt
from os.path import isfile

from ase.calculators.calculator import PropertyNotImplementedError
from ase.parallel import rank, barrier
from ase.io.trajectory import Trajectory
from ase.utils import basestring
import collections


class Dynamics:
    """Base-class for all MD and structure optimization classes."""
    def __init__(self, atoms, logfile, trajectory,
                 append_trajectory=False, master=None):
        """Dynamics object.

        Parameters:

        atoms: Atoms object
            The Atoms object to operate on.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory: Trajectory object or str
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* for no
            trajectory.

        append_trajectory: boolean
            Defaults to False, which causes the trajectory file to be
            overwriten each time the dynamics is restarted from scratch.
            If True, the new structures are appended to the trajectory
            file instead.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.
        """

        self.atoms = atoms
        if master is None:
            master = rank == 0
        if not master:
            logfile = None
        elif isinstance(logfile, basestring):
            if logfile == '-':
                logfile = sys.stdout
            else:
                logfile = open(logfile, 'a')
        self.logfile = logfile

        self.observers = []
        self.nsteps = 0

        if trajectory is not None:
            if isinstance(trajectory, basestring):
                mode = "a" if append_trajectory else "w"
                trajectory = Trajectory(trajectory, mode=mode,
                                        atoms=atoms, master=master)
            self.attach(trajectory)

    def get_number_of_steps(self):
        return self.nsteps

    def insert_observer(self, function, position=0, interval=1,
                        *args, **kwargs):
        """Insert an observer."""
        if not isinstance(function, collections.Callable):
            function = function.write
        self.observers.insert(position, (function, interval, args, kwargs))

    def attach(self, function, interval=1, *args, **kwargs):
        """Attach callback function.

        If *interval > 0*, at every *interval* steps, call *function* with
        arguments *args* and keyword arguments *kwargs*.

        If *interval <= 0*, after step *interval*, call *function* with
        arguments *args* and keyword arguments *kwargs*.  This is
        currently zero indexed."""

        if hasattr(function, 'set_description'):
            d = self.todict()
            d.update(interval=interval)
            function.set_description(d)
        if not hasattr(function, '__call__'):
            function = function.write
        self.observers.append((function, interval, args, kwargs))

    def call_observers(self):
        for function, interval, args, kwargs in self.observers:
            call = False
            # Call every interval iterations
            if interval > 0:
                if (self.nsteps % interval) == 0:
                    call = True
            # Call only on iteration interval
            elif interval <= 0:
                if self.nsteps == abs(interval):
                    call = True
            if call:
                function(*args, **kwargs)


class Optimizer(Dynamics):
    """Base-class for all structure optimization classes."""
    def __init__(self, atoms, restart, logfile, trajectory, master=None,
                 force_consistent=False):
        """Structure optimizer object.

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: str
            Filename for restart file.  Default value is *None*.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory: Trajectory object or str
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* for no
            trajectory.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K).  If force_consistent=None, uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.
        """
        Dynamics.__init__(self, atoms, logfile, trajectory, master)
        self.force_consistent = force_consistent
        self.restart = restart

        if restart is None or not isfile(restart):
            self.initialize()
        else:
            self.read()
            barrier()

    def todict(self):
        description = {'type': 'optimization',
                       'optimizer': self.__class__.__name__}
        return description

    def initialize(self):
        pass

    def irun(self, fmax=0.05, steps=100000000):
        """Run structure optimization algorithm as generator. This allows, e.g.,
        to easily run two optimizers at the same time.

        Examples:
        >>> opt1 = BFGS(atoms)
        >>> opt2 = BFGS(StrainFilter(atoms)).irun()
        >>> for _ in opt2:
        >>>     opt1.run()
        """

        if self.force_consistent is None:
            self.set_force_consistent()
        self.fmax = fmax
        step = 0
        while step < steps:
            f = self.atoms.get_forces()
            self.log(f)
            self.call_observers()
            if self.converged(f):
                yield True
                return
            self.step(f)
            yield False
            self.nsteps += 1
            step += 1

        yield False


    def run(self, fmax=0.05, steps=100000000):
        """Run structure optimization algorithm.

        This method will return when the forces on all individual
        atoms are less than *fmax* or when the number of steps exceeds
        *steps*.
        FloK: Move functionality into self.irun to be able to run as
              generator."""

        for converged in self.irun(fmax, steps):
            pass
        return converged

    def converged(self, forces=None):
        """Did the optimization converge?"""
        if forces is None:
            forces = self.atoms.get_forces()
        if hasattr(self.atoms, 'get_curvature'):
            return ((forces**2).sum(axis=1).max() < self.fmax**2 and
                    self.atoms.get_curvature() < 0.0)
        return (forces**2).sum(axis=1).max() < self.fmax**2

    def log(self, forces):
        fmax = sqrt((forces**2).sum(axis=1).max())
        e = self.atoms.get_potential_energy(
            force_consistent=self.force_consistent)
        T = time.localtime()
        if self.logfile is not None:
            name = self.__class__.__name__
            if self.nsteps == 0:
                self.logfile.write(
                    '%s  %4s %8s %15s %12s\n' %
                    (' ' * len(name), 'Step', 'Time', 'Energy', 'fmax'))
                if self.force_consistent:
                    self.logfile.write(
                        '*Force-consistent energies used in optimization.\n')
            self.logfile.write('%s:  %3d %02d:%02d:%02d %15.6f%1s %12.4f\n' %
                               (name, self.nsteps, T[3], T[4], T[5], e,
                                {1: '*', 0: ''}[self.force_consistent], fmax))
            self.logfile.flush()

    def dump(self, data):
        if rank == 0 and self.restart is not None:
            pickle.dump(data, open(self.restart, 'wb'), protocol=2)

    def load(self):
        return pickle.load(open(self.restart, 'rb'))

    def set_force_consistent(self):
        """Automatically sets force_consistent to True if force_consistent
        energies are supported by calculator; else False."""
        try:
            self.atoms.get_potential_energy(force_consistent=True)
        except PropertyNotImplementedError:
            self.force_consistent = False
        else:
            self.force_consistent = True
