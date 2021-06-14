from __future__ import print_function

# ******NOTICE***************
# optimize.py module by Travis E. Oliphant
#
# You may copy and use this module as you see fit with no
# guarantee implied provided you keep this notice in all copies.
# *****END NOTICE************

import time
import numpy as np
from numpy import eye, absolute, sqrt, isinf
from ase.utils.linesearch import LineSearch
from ase.optimize.optimize import Optimizer
from ase.utils import basestring


# These have been copied from Numeric's MLab.py
# I don't think they made the transition to scipy_core

# Modified from scipy_optimize
abs = absolute
pymin = min
pymax = max
__version__ = '0.1'


class BFGSLineSearch(Optimizer):
    def __init__(self, atoms, restart=None, logfile='-', maxstep=.2,
                 trajectory=None, c1=0.23, c2=0.46, alpha=10.0, stpmax=50.0,
                 master=None, force_consistent=None):
        """Optimize atomic positions in the BFGSLineSearch algorithm, which
        uses both forces and potential energy information.

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2 Angstroms).

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K).  By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.
        """
        self.maxstep = maxstep
        self.stpmax = stpmax
        self.alpha = alpha
        self.H = None
        self.c1 = c1
        self.c2 = c2
        self.force_calls = 0
        self.function_calls = 0
        self.r0 = None
        self.g0 = None
        self.e0 = None
        self.load_restart = False
        self.task = 'START'
        self.rep_count = 0
        self.p = None
        self.alpha_k = None
        self.no_update = False
        self.replay = False

        Optimizer.__init__(self, atoms, restart, logfile, trajectory,
                           master, force_consistent)

    def read(self):
        self.r0, self.g0, self.e0, self.task, self.H = self.load()
        self.load_restart = True

    def reset(self):
        print('reset')
        self.H = None
        self.r0 = None
        self.g0 = None
        self.e0 = None
        self.rep_count = 0

    def step(self, f):
        atoms = self.atoms
        from ase.neb import NEB
        if isinstance(atoms, NEB):
            raise TypeError('NEB calculations cannot use the BFGSLineSearch'
                            ' optimizer. Use BFGS or another optimizer.')
        r = atoms.get_positions()
        r = r.reshape(-1)
        g = -f.reshape(-1) / self.alpha
        p0 = self.p
        self.update(r, g, self.r0, self.g0, p0)
        # o,v = np.linalg.eigh(self.B)
        e = self.func(r)

        self.p = -np.dot(self.H, g)
        p_size = np.sqrt((self.p**2).sum())
        if p_size <= np.sqrt(len(atoms) * 1e-10):
            self.p /= (p_size / np.sqrt(len(atoms)*1e-10))
        ls = LineSearch()
        self.alpha_k, e, self.e0, self.no_update = \
            ls._line_search(self.func, self.fprime, r, self.p, g, e, self.e0,
                            maxstep=self.maxstep, c1=self.c1,
                            c2=self.c2, stpmax=self.stpmax)
        if self.alpha_k is None:
            raise RuntimeError("LineSearch failed!")

        dr = self.alpha_k * self.p
        atoms.set_positions((r + dr).reshape(len(atoms), -1))
        self.r0 = r
        self.g0 = g
        self.dump((self.r0, self.g0, self.e0, self.task, self.H))

    def update(self, r, g, r0, g0, p0):
        self.I = eye(len(self.atoms) * 3, dtype=int)
        if self.H is None:
            self.H = eye(3 * len(self.atoms))
            # self.B = np.linalg.inv(self.H)
            return
        else:
            dr = r - r0
            dg = g - g0
            # self.alpha_k can be None!!!
            if not (((self.alpha_k or 0) > 0 and
                    abs(np.dot(g, p0)) - abs(np.dot(g0, p0)) < 0) or
                    self.replay):
                return
            if self.no_update is True:
                print('skip update')
                return

            try:  # this was handled in numeric, let it remain for more safety
                rhok = 1.0 / (np.dot(dg, dr))
            except ZeroDivisionError:
                rhok = 1000.0
                print("Divide-by-zero encountered: rhok assumed large")
            if isinf(rhok):  # this is patch for np
                rhok = 1000.0
                print("Divide-by-zero encountered: rhok assumed large")
            A1 = self.I - dr[:, np.newaxis] * dg[np.newaxis, :] * rhok
            A2 = self.I - dg[:, np.newaxis] * dr[np.newaxis, :] * rhok
            self.H = (np.dot(A1, np.dot(self.H, A2)) +
                      rhok * dr[:, np.newaxis] * dr[np.newaxis, :])
            # self.B = np.linalg.inv(self.H)

    def func(self, x):
        """Objective function for use of the optimizers"""
        self.atoms.set_positions(x.reshape(-1, 3))
        self.function_calls += 1
        # Scale the problem as SciPy uses I as initial Hessian.
        return (self.atoms.get_potential_energy(
                force_consistent=self.force_consistent) / self.alpha)

    def fprime(self, x):
        """Gradient of the objective function for use of the optimizers"""
        self.atoms.set_positions(x.reshape(-1, 3))
        self.force_calls += 1
        # Remember that forces are minus the gradient!
        # Scale the problem as SciPy uses I as initial Hessian.
        f = self.atoms.get_forces().reshape(-1)
        return - f / self.alpha

    def replay_trajectory(self, traj):
        """Initialize hessian from old trajectory."""
        self.replay = True
        if isinstance(traj, basestring):
            from ase.io.trajectory import Trajectory
            traj = Trajectory(traj, 'r')
        r0 = None
        g0 = None
        for i in range(0, len(traj) - 1):
            r = traj[i].get_positions().ravel()
            g = - traj[i].get_forces().ravel() / self.alpha
            self.update(r, g, r0, g0, self.p)
            self.p = -np.dot(self.H, g)
            r0 = r.copy()
            g0 = g.copy()
        self.r0 = r0
        self.g0 = g0

    def log(self, forces):
        if self.logfile is None:
            return
        fmax = sqrt((forces**2).sum(axis=1).max())
        e = self.atoms.get_potential_energy(
            force_consistent=self.force_consistent)
        T = time.localtime()
        name = self.__class__.__name__
        w = self.logfile.write
        if self.nsteps == 0:
            w('%s  %4s[%3s] %8s %15s  %12s\n' %
              (' '*len(name), 'Step', 'FC', 'Time', 'Energy', 'fmax'))
            if self.force_consistent:
                w('*Force-consistent energies used in optimization.\n')
        w('%s:  %3d[%3d] %02d:%02d:%02d %15.6f%1s %12.4f\n'
            % (name, self.nsteps, self.force_calls, T[3], T[4], T[5], e,
               {1: '*', 0: ''}[self.force_consistent], fmax))
        self.logfile.flush()


def wrap_function(function, args):
    ncalls = [0]

    def function_wrapper(x):
        ncalls[0] += 1
        return function(x, *args)
    return ncalls, function_wrapper
